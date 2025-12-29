"""Evaluate ARES with MULTIPLE drift length training

Task modes:
    - matched: All drifts at default, prior matches, NOT trainable
    - misaligned: All drifts perturbed ±10-20%, prior wrong, IS trainable  
    - matched_prior_newtask: All drifts perturbed, prior correct, NOT trainable
"""

import os
import bo_cheetah_prior_ares
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
    vocs_config = """
        variables:
            q1: [-30, 30]
            q2: [-30, 30]
            cv: [-0.006, 0.006]
            q3: [-30, 30]
            ch:  [-0.006, 0.006]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Default beam
    default_beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
        mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
        sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
        sigma_y=torch.tensor(0.0002), sigma_py=torch.tensor(3.6941e-06),
        sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
        energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
    )

    # Mismatched beam
    mismatched_beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
        mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
        sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
        sigma_y=torch.tensor(0.0001), sigma_py=torch.tensor(3.6941e-06),
        sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
        energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
    )

    # Default drift lengths
    DEFAULT_DRIFTS = {
        "d1": 0.17504000663757324,
        "d2": 0.42800000309944153,
        "d3": 0.20399999618530273,
        "d4": 0.20399999618530273,
        "d5": 0.17900000512599945,
        "d6": 0.44999998807907104,
    }

    # Perturbed drifts (for misaligned tasks) - ±10-20% changes
    PERTURBED_DRIFTS = {
        "d1": 0.20,   # +14% from 0.175
        "d2": 0.50,   # +17% from 0.428
        "d3": 0.18,   # -12% from 0.204
        "d4": 0.24,   # +18% from 0.204
        "d5": 0.16,   # -11% from 0.179
        "d6": 0.52,   # +16% from 0.450
    }

    # Evaluator based on task
    if args.task == "matched":
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": None, "drift_lengths": DEFAULT_DRIFTS},
        )
    elif args.task == "misaligned":
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": mismatched_beam, "drift_lengths": PERTURBED_DRIFTS},
        )
    elif args.task == "matched_prior_newtask": 
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": mismatched_beam, "drift_lengths":  PERTURBED_DRIFTS},
        )
    else:
        raise ValueError(f"Invalid task: {args.task}")

    df = pd.DataFrame()

    for i in range(args.n_trials):
        print(f"\n{'='*70}")
        print(f"Trial {i+1}/{args.n_trials}")
        print(f"{'='*70}")

        if args.optimizer == "BO":
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)

        elif args.optimizer == "BO_prior":
            if args.task == "matched": 
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=None)
                # Set to default values
                prior_mean_module.d1 = DEFAULT_DRIFTS["d1"]
                prior_mean_module.d2 = DEFAULT_DRIFTS["d2"]
                prior_mean_module.d3 = DEFAULT_DRIFTS["d3"]
                prior_mean_module.d4 = DEFAULT_DRIFTS["d4"]
                prior_mean_module.d5 = DEFAULT_DRIFTS["d5"]
                prior_mean_module.d6 = DEFAULT_DRIFTS["d6"]
                gp_constructor = StandardModelConstructor(mean_modules={"mae": prior_mean_module})

            elif args.task == "misaligned":
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=None)  # WRONG beam
                # Set to WRONG default values (should learn PERTURBED)
                prior_mean_module.d1 = DEFAULT_DRIFTS["d1"]
                prior_mean_module.d2 = DEFAULT_DRIFTS["d2"]
                prior_mean_module.d3 = DEFAULT_DRIFTS["d3"]
                prior_mean_module.d4 = DEFAULT_DRIFTS["d4"]
                prior_mean_module.d5 = DEFAULT_DRIFTS["d5"]
                prior_mean_module.d6 = DEFAULT_DRIFTS["d6"]
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],  # TRAINABLE
                )

            elif args.task == "matched_prior_newtask":
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=mismatched_beam)  # CORRECT beam
                # Set to CORRECT perturbed values
                prior_mean_module.d1 = PERTURBED_DRIFTS["d1"]
                prior_mean_module.d2 = PERTURBED_DRIFTS["d2"]
                prior_mean_module.d3 = PERTURBED_DRIFTS["d3"]
                prior_mean_module.d4 = PERTURBED_DRIFTS["d4"]
                prior_mean_module.d5 = PERTURBED_DRIFTS["d5"]
                prior_mean_module.d6 = PERTURBED_DRIFTS["d6"]
                gp_constructor = StandardModelConstructor(mean_modules={"mae": prior_mean_module})

            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs, gp_constructor=gp_constructor)

        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        xopt = Xopt(vocs=vocs, evaluator=evaluator, generator=generator, max_evaluations=args.max_evaluation_steps)
        xopt.evaluate_data({"q1": 10.0, "q2": -10.0, "cv": 0.0, "q3": 10.0, "ch": 0.0})

        # Optimization loop
        for step_idx in tqdm.tqdm(range(args.max_evaluation_steps), desc="Optimizing"):
            xopt.step()

            # Debug for first trial
            if args.optimizer == "BO_prior" and i == 0 and step_idx < 5:  # Only first 5 steps
                try:
                    model = xopt.generator.model
                    gp_model = model.models[0]
                    learned_mean = gp_model.mean_module._model
                    drifts = learned_mean.get_drift_values()
                    
                    print(f"\n  Step {step_idx+1} - Drift lengths (m):")
                    for key in ["d1", "d2", "d3", "d4", "d5", "d6"]: 
                        print(f"    {key}: {drifts[key]:.6f}")
                except Exception as e:
                    print(f"  Debug error: {e}")

        # Post-processing
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Save learned drifts
        if args.optimizer == "BO_prior": 
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                drifts = learned_mean.get_drift_values()
                
                for key, value in drifts.items():
                    xopt.data[f"learned_{key}"] = value
                
                print(f"\n  Final learned drifts (m):")
                for key in ["d1", "d2", "d3", "d4", "d5", "d6"]:
                    print(f"    {key}: {drifts[key]:.6f}")
                
                if args.task in ["misaligned", "matched_prior_newtask"]:
                    print(f"\n  Ground truth (perturbed) drifts (m):")
                    for key in ["d1", "d2", "d3", "d4", "d5", "d6"]:
                        gt = PERTURBED_DRIFTS[key]
                        learned = drifts[key]
                        error = abs(learned - gt)
                        print(f"    {key}: {gt:.6f} (error: {error:.6f} m = {error/gt*100:.1f}%)")
                        xopt.data[f"error_{key}"] = error
            except Exception as e:
                print(f"  Warning: {e}")

        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}_multi_drift.csv"
    df.to_csv(out_filename)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED:  {out_filename}")
    print(f"{'='*70}")

    final_best_mae = df.groupby('run')['best_mae'].last()
    print(f"\nFinal Best MAE Summary:")
    print(f"  Mean ± Std: {final_best_mae.mean():.6e} ± {final_best_mae.std():.6e}")

    if args.optimizer == "BO_prior" and args.task == "misaligned":
        print(f"\n{'='*70}")
        print(f"Multi-Parameter Learning Summary:")
        print(f"{'='*70}")
        for key in ["d1", "d2", "d3", "d4", "d5", "d6"]:
            if f"learned_{key}" in df.columns:
                final_learned = df.groupby('run')[f'learned_{key}'].last()
                gt = PERTURBED_DRIFTS[key]
                print(f"\n{key.upper()}:")
                print(f"  Ground truth: {gt:.6f} m")
                print(f"  Learned: {final_learned.mean():.6f} ± {final_learned.std():.6f} m")
                if f"error_{key}" in df.columns:
                    final_errors = df.groupby('run')[f'error_{key}'].last()
                    print(f"  Mean error: {final_errors.mean():.6f} m ({final_errors.mean()/gt*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ARES with multiple drift training")
    parser.add_argument("--optimizer", type=str, default="BO_prior", choices=["BO", "BO_prior", "NM"])
    parser.add_argument("--n_trials", "-n", type=int, default=5)
    parser.add_argument("--task", "-t", type=str, default="misaligned", 
                        choices=["matched", "misaligned", "matched_prior_newtask"])
    parser.add_argument("--max_evaluation_steps", "-s", type=int, default=100)
    parser.add_argument("--output_dir", "-o", type=str, default="data_ares/")
    args = parser.parse_args()
    main(args)