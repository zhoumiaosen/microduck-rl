"""MicroDuck spin terms."""

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch
from .common import (
    _DEFAULT_ASSET_CFG,
)


# --------------------------------------------------------------------------- #
# Tâche SPIN — rotation rapide sur place sur rollers                            #
# --------------------------------------------------------------------------- #
# Enveloppe de phase : la commande du slot bouton porte une phase, qui pilote
# une VITESSE DE LACET cible en trapèze (et non une pose comme le crouch).
#   [0, accel_end)        0.5 s   0 -> rate_max    (lancement)
#   [accel_end, hold_end) 1.6 s   rate_max         (régime)
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    (freinage)
#   [brake_end, 1.0)      1.4 s   0                (repos debout)
# Aire sous l'enveloppe sur un cycle = 2.1 * SPIN_RATE_MAX rad. À 3.0 rad/s :
# 2.1 * 3.0 = 6.3 rad ~ 1 tour (et non ~2, comme avec l'ancienne cible 6.0).
SPIN_PERIOD = 4.0


SPIN_RATE_MAX = 3.0


SPIN_ACCEL_END = 0.125


SPIN_HOLD_END = 0.525


SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Vitesse de lacet cible (rad/s, positive = anti-horaire) le long de la phase."""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(
        brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w
    )
    return w


def spin_gate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Porte de shaping dans [0,1] = enveloppe normalisée.

    Vaut 0 sur tout le segment de repos : les amorces (ciseau des jambes,
    différentiel des roues) ne s'appliquent que pendant lancement + régime, donc
    le robot revient en station neutre avant de rendre la main à la policy roller.
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """Récupère la phase [0,1) depuis la commande [cos(2πφ), sin(2πφ), 0] du slot."""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(
    omega_z: torch.Tensor, omega_target: torch.Tensor, std: float
) -> torch.Tensor:
    """Gaussienne sur l'erreur de vitesse de lacet (fonction pure, testable)."""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 1.5,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Objectif principal du spin : suivre la vitesse de lacet cible ω*(φ).

    ω_z est pris en repère corps (c'est ce que voit le gyro de l'IMU, donc ce que
    la policy observe). Une rotation dans le mauvais sens est plus punie que
    l'immobilité, la gaussienne étant centrée sur une cible positive.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 : gradient constant vers la cible même quand la gaussienne
    de `spin_rate_track` sature loin de la cible. À utiliser avec un poids
    POSITIF (la valeur retournée est déjà négative)."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # atténuation du coût de dérive pendant le lancement


def spin_stay_in_place(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE,
    accel_end: float = SPIN_ACCEL_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Coût ‖v_xy‖² du tronc : tourner SUR PLACE, et tuer l'élan d'entrée.

    Pas d'état de référence (contrairement à une dérive mesurée depuis le reset),
    donc reste valide sur les 5 cycles d'un épisode. À utiliser avec un poids
    NÉGATIF.

    ATTÉNUÉ PENDANT LE LANCEMENT : sur `[0, accel_end)` le robot doit pousser au
    sol pour s'injecter du moment angulaire, et l'état d'entrée lui donne jusqu'à
    0.3 m/s qu'il est censé CONVERTIR en rotation. Facturer la translation à plein
    tarif à cet instant s'oppose donc directement à l'objectif. Le coût est
    multiplié par `launch_scale` sur ce seul segment, et vaut plein tarif ensuite
    (régime, freinage, repos) où « sur place » est le vrai critère.

    Contrairement aux autres amorces du spin, ce terme n'est PAS éteint par
    `spin_gate_by_phase` : pendant le repos on veut justement qu'il reste plein,
    puisque c'est là que le robot doit être immobile.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(
        phase < accel_end,
        torch.full_like(cost, launch_scale),
        torch.ones_like(cost),
    )
    return cost * scale


# Demi-voie mesurée sur le modèle rollers (pose HOME, sites left_foot/right_foot) :
# 0.0499 m, contre 0.03 m estimé au spec. Conséquence mécanique de SPIN_RATE_MAX
# (A1) : différentiel attendu = 2*SPIN_RATE_MAX*demi_voie/r, r = 0.0175 m.
# À l'ancienne cible 6.0 rad/s : 2*6.0*0.0499/0.0175 = 34.2 rad/s (retenu comme
# 34.0, soit +71% par rapport aux 20.0 estimés au spec -> seuil de 30% dépassé).
# À la nouvelle cible 3.0 rad/s : 2*3.0*0.0499/0.0175 = 17.1 rad/s. Laisser 34.0
# ici plafonnerait le terme à tanh(17.1/34) = 0.47 de son propre maximum, ce qui
# affaiblirait exactement le shaping qu'on veut renforcer (cf. spin_stay_in_place).
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s ; recalibré sur la demi-voie mesurée et SPIN_RATE_MAX = 3.0


def spin_wheel_differential_from_values(
    diff: torch.Tensor, gate: torch.Tensor, omega_scale: float
) -> torch.Tensor:
    """Fonction pure : tanh du différentiel de roues, portée par gate, clampée ≥ 0."""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    omega_scale: float = SPIN_WHEEL_OMEGA_SCALE,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Récompense la rotation EN ROULEMENT (et non en patinage).

    Pour un spin anti-horaire, le patin gauche recule et le droit avance ; les 4
    roues tournant positif en marche avant, cela donne ω_D − ω_G > 0. Le tanh
    sature à `omega_scale` pour éviter la course à la vitesse de roue.
    """
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(
        omega_right - omega_left, gate, omega_scale
    )


def spin_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Les deux lames au sol pendant le spin — empêche « je saute et je vrille ».

    Variante de `grounded_reward` du swizzle, qui n'est pas réutilisable ici :
    elle se pondère par cmd_x, qui vaut cos(2πφ) sur la commande de phase.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_pitch", "knee"),
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Amorce le CISEAU des jambes (une avant / une arrière) pendant le spin.

    Le robot a des conventions de signe MIROIR gauche/droite : une pose
    symétrique satisfait q_G + q_D ≈ 0 (cf. `leg_symmetry_reward`), donc le
    ciseau satisfait q_G ≈ q_D. On retourne `gate(φ) · (−mean|q_G − q_D|)` — à
    utiliser avec un poids POSITIF, décroissant par curriculum : l'amorce
    s'efface pour laisser la policy affiner son propre geste.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor
