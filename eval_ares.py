"""Evaluate ARES with drift length training (FODO-style three scenarios)

Task modes (EXACTLY matching FODO):
    - matched: Prior drift matches task (0.428m), default beam, prior NOT trainable
    - misaligned: Prior drift wrong (0.428m vs 0.6m), different beam, prior IS trainable
    - matched_prior_newtask: Prior drift correct (0.6m), different beam, prior NOT trainable
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
            ch: [-0.006, 0.006]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Default beam (for matched task)
    default_beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
        mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
        sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
        sigma_y=torch.tensor(0.0002), sigma_py=torch.tensor(3.6941e-06),
        sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
        energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
    )

    # Mismatched beam (for misaligned tasks - different sigma_y)
    mismatched_beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
        mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
        sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
        sigma_y=torch.tensor(0.0001),  # DIFFERENT
        sigma_py=torch.tensor(3.6941e-06),
        sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
        energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
    )

    # Evaluator based on task (EXACTLY like FODO)
    if args.task == "matched": 
        # MATCHED: Default beam, drift = 0.428m (matches prior)
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": incoming_beam, "drift_length": 0.428},
        )
    elif args.task == "misaligned":
        # MISALIGNED: Different beam, drift = 0.6m (different from prior's 0.428m)
        incoming_beam = mismatched_beam
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": incoming_beam, "drift_length":  0.6},
        )
    elif args.task == "matched_prior_newtask": 
        # MATCHED_PRIOR_NEWTASK: Different beam, drift = 0.6m (prior will be set to 0.6m)
        incoming_beam = mismatched_beam
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_lattice_problem,
            function_kwargs={"incoming_beam": incoming_beam, "drift_length": 0.6},
        )
    else:
        raise ValueError(f"Invalid task: {args.task}")

    df = pd.DataFrame()

    for i in range(args.n_trials):
        print(f"\n{'='*70}")
        print(f"Trial {i+1}/{args.n_trials}")
        print(f"{'='*70}")

        # Initialize Generator (EXACTLY like FODO)
        if args.optimizer == "BO": 
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)

        elif args.optimizer == "BO_prior":
            if args.task == "matched":
                # MATCHED: Prior matches task perfectly
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=None)
                prior_mean_module.drift_length = 0.428  # Matches evaluation
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )

            elif args.task == "misaligned":
                # MISALIGNED: Prior is wrong, needs to learn
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=None)  # WRONG beam
                prior_mean_module.drift_length = 0.428  # WRONG drift (should be 0.6)
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],  # TRAINABLE
                )

            elif args.task == "matched_prior_newtask": 
                # MATCHED_PRIOR_NEWTASK: Prior pre-configured correctly
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(incoming_beam=mismatched_beam)  # CORRECT beam
                prior_mean_module.drift_length = 0.6  # CORRECT drift
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )

            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs, gp_constructor=gp_constructor)

        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        xopt = Xopt(vocs=vocs, evaluator=evaluator, generator=generator, max_evaluations=args.max_evaluation_steps)
        
        # Fixed starting point
        xopt.evaluate_data({"q1": 10.0, "q2": -10.0, "cv": 0.0, "q3":  10.0, "ch":  0.0})

        # Optimization loop
        for step_idx in tqdm.tqdm(range(args.max_evaluation_steps), desc="Optimizing"):
            xopt.step()

            # Debug output for first trial
            if args.optimizer == "BO_prior" and i == 0:
                try:
                    model = xopt.generator.model
                    gp_model = model.models[0]
                    learned_mean = gp_model.mean_module._model
                    drift = learned_mean.get_drift_length_value()
                    
                    print(f"\n  Step {step_idx+1}:")
                    print(f"    drift_length: {drift:.6f} m (raw: {learned_mean.raw_drift_length.data.item():.6f})")
                except Exception as e:
                    print(f"  Debug error: {e}")

        # Post-processing
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Save learned drift length
        if args.optimizer == "BO_prior":
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                drift = learned_mean.get_drift_length_value()
                xopt.data["learned_drift_length"] = drift
                
                print(f"\n  Final learned drift_length: {drift:.6f} m")
                
                if args.task == "misaligned":
                    print(f"  Ground truth drift_length: 0.600000 m")
                    error = abs(drift - 0.6)
                    print(f"  Learning error: {error:.6f} m ({error/0.6*100:.2f}%)")
                elif args.task == "matched_prior_newtask":
                    print(f"  Initial drift_length was set to 0.6 m (correct)")
            except Exception as e:
                print(f"  Warning: {e}")

        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}_drift.csv"
    df.to_csv(out_filename)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED:  {out_filename}")
    print(f"{'='*70}")

    # Summary statistics
    final_best_mae = df.groupby('run')['best_mae'].last()
    print(f"\nFinal Best MAE Summary:")
    print(f"  Mean ± Std: {final_best_mae.mean():.6e} ± {final_best_mae.std():.6e}")

    if args.optimizer == "BO_prior" and "learned_drift_length" in df.columns:
        final_drifts = df.groupby('run')['learned_drift_length'].last()
        print(f"\nLearned Drift Length Summary:")
        print(f"  Mean ± Std: {final_drifts.mean():.6f} ± {final_drifts.std():.6f} m")
        print(f"  Range: {final_drifts.min():.6f} to {final_drifts.max():.6f} m")
        
        if args.task == "misaligned":
            print(f"  Ground truth:  0.600000 m")
            print(f"  Mean error: {abs(final_drifts.mean() - 0.6):.6f} m")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ARES drift length test (FODO-style)")
    parser.add_argument("--optimizer", type=str, default="BO", choices=["BO", "BO_prior", "NM"])
    parser.add_argument("--n_trials", "-n", type=int, default=10)
    parser.add_argument("--task", "-t", type=str, default="matched", 
                        choices=["matched", "misaligned", "matched_prior_newtask"])
    parser.add_argument("--max_evaluation_steps", "-s", type=int, default=50)
    parser.add_argument("--output_dir", "-o", type=str, default="data_ares/")
    args = parser.parse_args()
    main(args)