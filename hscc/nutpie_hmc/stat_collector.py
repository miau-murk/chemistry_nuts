import arviz as az

class Statistic(az.data.inference_data.InferenceData):
    def __init__(self, inference_data=None):
        """Initialize with optional InferenceData

        Args:
            inference_data: Existing InferenceData object to wrap
        """
        if inference_data is not None:
            # copy InferenceData
            self.__dict__.update(inference_data.__dict__)
        self._init_stats_lists()

    def _init_stats_lists(self):
        """Initialize statistic category lists"""
        self.basic_stats = [
            "depth",
            "maxdepth_reached",
            "logp",
            "energy",
            "diverging",
            "step_size",
            "step_size_bar",
            "n_steps",
        ]

        self.detailed_stats = [
            "gradient",
            "unconstrained_draw",
            "divergence_start",
            "divergence_end",
            "divergence_momentum",
        ]

    def save_to_log(self, filename: str = "sampling_stats.log") -> None:
        """Save statistics to text file"""
        if not hasattr(self, "sample_stats"):
            raise ValueError("No sampling data available")

        with open(filename, "w") as f:
            f.write("Sampling Statistics Report\n")
            f.write("=" * 30 + "\n")
            self._save_basic_stats(f)

    def _save_basic_stats(self, file_obj):
        """Helper method to save basic stats with aligned columns"""
        # saving samples
        file_obj.write("\n\nGenerated Samples:\n")
        samples = self.posterior.y.values  # (chains, draws, dim)
        n_chains, n_samples, n_dim = samples.shape

        header = "Chain | Step"
        file_obj.write(header + "\n")
        file_obj.write("-" * len(header) + "\n")

        for chain in range(n_chains):
            for step in range(n_samples):
                sample_values = samples[chain, step, :]
                row = f"{chain} | {step + 1} " + " ".join(
                    f"{val:.4f}" for val in sample_values
                )
                file_obj.write(row + "\n")

        # saving statistics
        file_obj.write("\nBasic Statistics:\n")
        file_obj.write("-" * 20 + "\n")

        stats = self.sample_stats
        n_chains = len(stats.chain)
        n_steps = len(stats.draw)

        col_widths = {
            "Step": 6,
            "depth": 10,
            "maxdepth_reached": 16,
            "logp": 10,
            "energy": 10,
            "diverging": 11,
            "step_size": 11,
            "step_size_bar": 14,
            "n_steps": 10,
        }

        header_fmt = " ".join(f"{{:<{width}}}" for width in col_widths.values())
        row_fmt = " ".join(
            (f"{{:<{width}}}" if name == "Step" else f"{{:<{width}.4f}}")
            for name, width in col_widths.items()
        )

        for chain in range(n_chains):
            file_obj.write(f"\nChain {chain}\n")
            file_obj.write(header_fmt.format(*col_widths.keys()) + "\n")
            file_obj.write(
                "-" * (sum(col_widths.values()) + 3 * (len(col_widths) - 1)) + "\n"
            )

            for step in range(n_steps):
                row_data = {"Step": str(step + 1)}
                for stat_name in self.basic_stats:
                    if hasattr(stats, stat_name):
                        values = getattr(stats, stat_name).values
                        row_data[stat_name] = values[chain, step]
                    else:
                        row_data[stat_name] = float("nan")

                formatted_values = []
                for name in col_widths:
                    val = row_data[name]
                    if name == "Step":
                        formatted_values.append(val)
                    else:
                        formatted_values.append(float(val))

                file_obj.write(row_fmt.format(*formatted_values) + "\n")
