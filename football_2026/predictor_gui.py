"""Simple Tkinter GUI for World Cup Predictor.

Usage:
    python predictor_gui.py

Features:
- View top 10 championship probabilities
- Add match results interactively
- See updated predictions in real-time
- View backtest metrics
"""
from datetime import date
from pathlib import Path
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from predictor import WorldCupPredictor, OUT_DIR


class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2026 World Cup Predictor")
        self.root.geometry("900x700")

        self.predictor = WorldCupPredictor(n_simulations=3000)
        self.results = None

        self._build_ui()

    def _build_ui(self):
        # Control panel
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Load actuals
        ttk.Label(control_frame, text="CSV Path:").grid(row=0, column=0, sticky="w")
        self.csv_path = ttk.Entry(control_frame, width=40)
        self.csv_path.insert(0, "out/actual_results_2026.csv")
        self.csv_path.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Load Results", command=self.load_actuals).grid(
            row=0, column=2, padx=5
        )

        ttk.Button(control_frame, text="Run Predictions", command=self.run_predictions).grid(
            row=0, column=3, padx=5
        )

        # Add result panel
        add_frame = ttk.LabelFrame(self.root, text="Add Match Result", padding=10)
        add_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(add_frame, text="Date:").grid(row=0, column=0, sticky="w")
        self.date_var = ttk.Entry(add_frame, width=12)
        self.date_var.insert(0, "2026-06-12")
        self.date_var.grid(row=0, column=1, padx=2)

        ttk.Label(add_frame, text="Home:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.home_team = ttk.Entry(add_frame, width=15)
        self.home_team.grid(row=0, column=3, padx=2)

        ttk.Label(add_frame, text="Away:").grid(row=0, column=4, sticky="w")
        self.away_team = ttk.Entry(add_frame, width=15)
        self.away_team.grid(row=0, column=5, padx=2)

        ttk.Label(add_frame, text="HScore:").grid(row=0, column=6, sticky="w")
        self.home_score = ttk.Entry(add_frame, width=3)
        self.home_score.insert(0, "1")
        self.home_score.grid(row=0, column=7, padx=2)

        ttk.Label(add_frame, text="AScore:").grid(row=0, column=8, sticky="w")
        self.away_score = ttk.Entry(add_frame, width=3)
        self.away_score.insert(0, "0")
        self.away_score.grid(row=0, column=9, padx=2)

        ttk.Button(add_frame, text="Add Result", command=self.add_result).grid(
            row=0, column=10, padx=10
        )

        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Top 10 Championship Probabilities", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = ScrolledText(results_frame, height=15, font=("Consolas", 10))
        self.results_text.pack(fill="both", expand=True)

        # Group matches
        group_frame = ttk.LabelFrame(self.root, text="Remaining Group Matches", padding=10)
        group_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.group_text = ScrolledText(group_frame, height=12, font=("Consolas", 9))
        self.group_text.pack(fill="both", expand=True)

    def load_actuals(self):
        csv_path = self.csv_path.get()
        if Path(csv_path).exists():
            try:
                n = self.predictor.load_actuals(csv_path)
                messagebox.showinfo("Success", f"Loaded {n} results from {csv_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Error", f"File not found: {csv_path}")

    def add_result(self):
        try:
            d = pd.Timestamp(self.date_var.get()).date()
            h = self.home_team.get()
            a = self.away_team.get()
            hs = int(self.home_score.get())
            as_ = int(self.away_score.get())
            self.predictor.add_result(d, h, a, hs, as_)
            self.home_team.delete(0, "end")
            self.away_team.delete(0, "end")
            messagebox.showinfo("Success", f"Added: {h} {hs}-{as_} {a}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_predictions(self):
        with self.root.after(100):
            self.results = self.predictor.predict_tournament()
        self._display_results()

    def _display_results(self):
        if not self.results:
            return

        # Top 10
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "-" * 50 + "\n")

        if self.results.backtest_metrics and self.results.backtest_metrics.get("n_matches", 0) > 0:
            m = self.results.backtest_metrics
            self.results_text.insert(
                "end",
                f"Backtest: {m['n_matches']} matches, "
                f"{m['outcome_accuracy']*100:.1f}% accuracy, "
                f"Brier={m['brier']:.3f}\n"
            )

        self.results_text.insert("end", "-" * 50 + "\n")

        if self.results.tournament_probs is not None:
            for _, row in self.results.tournament_probs.head(10).iterrows():
                self.results_text.insert(
                    "end",
                    f"{row['P_Win']*100:5.1f}%  {row['team']:<20s} (ELO {row['elo']:.0f})\n"
                )

        # Group matches
        self.group_text.delete("1.0", "end")
        if self.results.group_match_predictions is not None:
            for letter in sorted(set(self.results.group_match_predictions["group"])):
                matches = self.results.group_match_predictions[
                    self.results.group_match_predictions["group"] == letter
                ]
                for _, r in matches.iterrows():
                    self.group_text.insert(
                        "end",
                        f"{letter} {r['home'][:15]:15s} vs {r['away'][:15]:15s}  "
                        f"{r['ph']:.2f}/{r['pd']:.2f}/{r['pa']:.2f}\n"
                    )


if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = PredictorGUI(root)
    root.mainloop()