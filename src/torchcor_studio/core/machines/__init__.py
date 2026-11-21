"""Compute machines: this computer and SSH hosts behind one interface."""

from torchcor_studio.core.machines.base import Machine, MachineError

__all__ = ["Machine", "MachineError", "create_machine"]


def create_machine(profile):  # noqa: ANN001, ANN201 - thin factory
    """Instantiate the right :class:`Machine` for a profile."""
    from torchcor_studio.core.models import MachineKind

    if profile.kind == MachineKind.LOCAL:
        from torchcor_studio.core.machines.local import LocalMachine

        return LocalMachine(profile)
    from torchcor_studio.core.machines.ssh import SSHMachine

    return SSHMachine(profile)
