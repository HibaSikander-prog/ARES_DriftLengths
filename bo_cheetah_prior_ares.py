"""TEST:  Train drift length like FODO to verify training mechanism works

Three scenarios matching FODO:
1.matched: Prior drift matches task drift (both 0.428m), prior NOT trainable
2.misaligned: Prior drift wrong (0.428m vs 0.6m), beam wrong, prior IS trainable
3.matched_prior_newtask: Prior drift correct (0.6m), beam correct, prior NOT trainable
"""

from typing import Dict, Optional
import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Positive
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


def ares_lattice_problem(
    input_param: Dict[str, float],
    incoming_beam:  Optional[cheetah.Beam] = None,
    drift_length: float = 0.428,  # Configurable D2 drift length
) -> Dict[str, float]: 
    """ARES lattice with configurable D2 drift length (like FODO drift)"""
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0002), sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
        )
    
    # Fixed lengths
    d1_length = torch.tensor(0.17504000663757324)
    d2_length = torch.tensor(drift_length)  # VARIABLE (like FODO drift)
    d3_length = torch.tensor(0.20399999618530273)
    d4_length = torch.tensor(0.20399999618530273)
    d5_length = torch.tensor(0.17900000512599945)
    d6_length = torch.tensor(0.44999998807907104)
    quad_length = torch.tensor(0.12200000137090683)
    corrector_length = torch.tensor(0.019999999552965164)
    
    ares_segment = cheetah.Segment([
        cheetah.Drift(length=d1_length, name="D1"),
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q1"]), name="Q1"),
        cheetah.Drift(length=d2_length, name="D2"),  # TRAINABLE DRIFT
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q2"]), name="Q2"),
        cheetah.Drift(length=d3_length, name="D3"),
        cheetah.VerticalCorrector(length=corrector_length, angle=torch.tensor(input_param["cv"]), name="CV"),
        cheetah.Drift(length=d4_length, name="D4"),
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q3"]), name="Q3"),
        cheetah.Drift(length=d5_length, name="D5"),
        cheetah.HorizontalCorrector(length=corrector_length, angle=torch.tensor(input_param["ch"]), name="CH"),
        cheetah.Drift(length=d6_length, name="D6"),
    ])
    
    out_beam = ares_segment(incoming_beam)
    beam_quality_mse = out_beam.mu_x**2 + out_beam.mu_y**2 + 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_quality_mae = out_beam.mu_x.abs() + out_beam.mu_y.abs() + 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    
    return {
        "mse":  beam_quality_mse.detach().numpy(),
        "log_mse": beam_quality_mse.log().detach().numpy(),
        "mae": beam_quality_mae.detach().numpy(),
        "log_mae": beam_quality_mae.log().detach().numpy(),
        "mu_x": out_beam.mu_x.detach().numpy(),
        "mu_y": out_beam.mu_y.detach().numpy(),
        "sigma_x": out_beam.sigma_x.detach().numpy(),
        "sigma_y": out_beam.sigma_y.detach().numpy(),
    }


class AresPriorMean(Mean):
    """ARES prior with trainable drift_length (exactly like FODO)"""

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
        super().__init__()
        
        if incoming_beam is None:
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
                mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
                sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
                sigma_y=torch.tensor(0.0002), sigma_py=torch.tensor(3.6941e-06),
                sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
                energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
            )
        self.incoming_beam = incoming_beam
        
        # Fixed lengths
        d1_length = torch.tensor(0.17504000663757324)
        d3_length = torch.tensor(0.20399999618530273)
        d4_length = torch.tensor(0.20399999618530273)
        d5_length = torch.tensor(0.17900000512599945)
        d6_length = torch.tensor(0.44999998807907104)
        quad_length = torch.tensor(0.12200000137090683)
        corrector_length = torch.tensor(0.019999999552965164)
        
        # Build lattice
        self.D1 = cheetah.Drift(length=d1_length, name="D1")
        self.Q1 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q1")
        self.D2 = cheetah.Drift(length=torch.tensor([0.428]), name="D2")  # Will be updated
        self.Q2 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q2")
        self.D3 = cheetah.Drift(length=d3_length, name="D3")
        self.CV = cheetah.VerticalCorrector(length=corrector_length, angle=torch.tensor([0.0]), name="CV")
        self.D4 = cheetah.Drift(length=d4_length, name="D4")
        self.Q3 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q3")
        self.D5 = cheetah.Drift(length=d5_length, name="D5")
        self.CH = cheetah.HorizontalCorrector(length=corrector_length, angle=torch.tensor([0.0]), name="CH")
        self.D6 = cheetah.Drift(length=d6_length, name="D6")
        
        self.segment = cheetah.Segment([
            self.D1, self.Q1, self.D2, self.Q2, self.D3,
            self.CV, self.D4, self.Q3, self.D5, self.CH, self.D6
        ])
        
        # Trainable drift length (EXACTLY like FODO)
        drift_length_constraint = Positive()
        self.register_parameter("raw_drift_length", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "drift_length_prior",
            SmoothedBoxPrior(0.2, 0.8),
            self._drift_length_param,
            self._set_drift_length,
        )
        self.register_constraint("raw_drift_length", drift_length_constraint)

    def _drift_length_param(self, m):
        return m.raw_drift_length_constraint.transform(m.raw_drift_length)
    
    def _set_drift_length(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_drift_length)
        m.initialize(raw_drift_length=m.raw_drift_length_constraint.inverse_transform(value))
    
    @property
    def drift_length(self):
        return self._drift_length_param(self)
    
    @drift_length.setter
    def drift_length(self, value):
        self._set_drift_length(self, value)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.CV.angle = X[..., 2]
        self.Q3.k1 = X[..., 3]
        self.CH.angle = X[..., 4]
        
        # Update D2 with trainable drift length
        self.D2.length = self.drift_length
        
        out_beam = self.segment(self.incoming_beam)
        beam_quality_mae = out_beam.mu_x.abs() + out_beam.mu_y.abs() + 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
        
        return beam_quality_mae

    def get_drift_length_value(self):
        drift_val = self.drift_length
        if torch.is_tensor(drift_val):
            return float(drift_val.item())
        return float(drift_val)