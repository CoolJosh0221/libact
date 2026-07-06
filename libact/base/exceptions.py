"""Custom exceptions for libact."""


class ExtensionUnavailable(ImportError):
    """Raised when an optional compiled C extension is not available.

    libact ships two query strategies --
    :class:`~libact.query_strategies.hintsvm.HintSVM` and
    :class:`~libact.query_strategies.variance_reduction.VarianceReduction` --
    that are backed by compiled C/C++ extensions linking BLAS/LAPACK. These
    extensions are only built on POSIX platforms (Linux and macOS); there is no
    Windows build. On Windows, or on any install where the extensions were not
    compiled, importing these strategy classes still succeeds, but instantiating
    them raises this error.

    It subclasses :class:`ImportError` so that existing ``except ImportError``
    guards keep working.
    """

    @classmethod
    def for_strategy(cls, strategy_name, module_name):
        """Build an :class:`ExtensionUnavailable` with a helpful message.

        Parameters
        ----------
        strategy_name : str
            The public strategy class name (e.g. ``'HintSVM'``).
        module_name : str
            The private compiled module that could not be imported
            (e.g. ``'libact.query_strategies._hintsvm'``).
        """
        return cls(
            "{strategy} requires the '{module}' C extension, which is not "
            "available in this installation. This extension links BLAS/LAPACK "
            "and is only built on POSIX platforms (Linux and macOS); there is "
            "no Windows build. To use {strategy}, install libact on Linux or "
            "macOS with BLAS/LAPACK available, or on Windows run it under WSL. "
            "The pure-Python query strategies (e.g. UncertaintySampling, "
            "QueryByCommittee, QUIRE) work on every platform. See the README "
            "'Build Options' section for details.".format(
                strategy=strategy_name, module=module_name
            )
        )
