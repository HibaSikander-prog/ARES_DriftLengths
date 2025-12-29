"""TEST: Train ALL 6 drift lengths to verify multi-parameter training works

Three scenarios:
1.matched: All drifts at default values, prior matches, NOT trainable
2.misaligned: All drifts perturbed ±10-20%, prior wrong, IS trainable
3.matched_prior_newtask: All drifts perturbed, prior correct, NOT trainable
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
    drift_lengths: Optional[Dict[str, float]] = None,
) -> Dict[str, float]: 
    """ARES lattice with configurable drift lengths
    
    Args:
        input_param:  Magnet settings (q1, q2, cv, q3, ch)
        incoming_beam: Beam parameters
        drift_lengths: Dict with keys d1, d2, d3, d4, d5, d6
    """
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07), mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06), mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002), sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0002), sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06), sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08), total_charge=torch.tensor(5.0000e-13),
        )
    
    # Default drift lengths
    if drift_lengths is None:
        drift_lengths = {
            "d1": 0.17504000663757324,
            "d2": 0.42800000309944153,
            "d3": 0.20399999618530273,
            "d4": 0.20399999618530273,
            "d5": 0.17900000512599945,
            "d6": 0.44999998807907104,
        }
    
    quad_length = torch.tensor(0.12200000137090683)
    corrector_length = torch.tensor(0.019999999552965164)
    
    # Build segment with configurable drifts
    ares_segment = cheetah.Segment([
        cheetah.Drift(length=torch.tensor(drift_lengths["d1"]), name="D1"),
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q1"]), name="Q1"),
        cheetah.Drift(length=torch.tensor(drift_lengths["d2"]), name="D2"),
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q2"]), name="Q2"),
        cheetah.Drift(length=torch.tensor(drift_lengths["d3"]), name="D3"),
        cheetah.VerticalCorrector(length=corrector_length, angle=torch.tensor(input_param["cv"]), name="CV"),
        cheetah.Drift(length=torch.tensor(drift_lengths["d4"]), name="D4"),
        cheetah.Quadrupole(length=quad_length, k1=torch.tensor(input_param["q3"]), name="Q3"),
        cheetah.Drift(length=torch.tensor(drift_lengths["d5"]), name="D5"),
        cheetah.HorizontalCorrector(length=corrector_length, angle=torch.tensor(input_param["ch"]), name="CH"),
        cheetah.Drift(length=torch.tensor(drift_lengths["d6"]), name="D6"),
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
    """ARES prior with 6 trainable drift lengths"""

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
        
        quad_length = torch.tensor(0.12200000137090683)
        corrector_length = torch.tensor(0.019999999552965164)
        
        # Build lattice with placeholder drifts
        self.D1 = cheetah.Drift(length=torch.tensor([0.175]), name="D1")
        self.Q1 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q1")
        self.D2 = cheetah.Drift(length=torch.tensor([0.428]), name="D2")
        self.Q2 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q2")
        self.D3 = cheetah.Drift(length=torch.tensor([0.204]), name="D3")
        self.CV = cheetah.VerticalCorrector(length=corrector_length, angle=torch.tensor([0.0]), name="CV")
        self.D4 = cheetah.Drift(length=torch.tensor([0.204]), name="D4")
        self.Q3 = cheetah.Quadrupole(length=quad_length, k1=torch.tensor([0.0]), name="Q3")
        self.D5 = cheetah.Drift(length=torch.tensor([0.179]), name="D5")
        self.CH = cheetah.HorizontalCorrector(length=corrector_length, angle=torch.tensor([0.0]), name="CH")
        self.D6 = cheetah.Drift(length=torch.tensor([0.450]), name="D6")
        
        self.segment = cheetah.Segment([
            self.D1, self.Q1, self.D2, self.Q2, self.D3,
            self.CV, self.D4, self.Q3, self.D5, self.CH, self.D6
        ])
        
        # ===== 6 TRAINABLE DRIFT LENGTHS =====
        drift_constraint = Positive()
        
        # D1
        self.register_parameter("raw_d1", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d1_prior", SmoothedBoxPrior(0.1, 0.3), self._d1_param, self._set_d1)
        self.register_constraint("raw_d1", drift_constraint)
        
        # D2
        self.register_parameter("raw_d2", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d2_prior", SmoothedBoxPrior(0.3, 0.6), self._d2_param, self._set_d2)
        self.register_constraint("raw_d2", drift_constraint)
        
        # D3
        self.register_parameter("raw_d3", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d3_prior", SmoothedBoxPrior(0.1, 0.3), self._d3_param, self._set_d3)
        self.register_constraint("raw_d3", drift_constraint)
        
        # D4
        self.register_parameter("raw_d4", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d4_prior", SmoothedBoxPrior(0.1, 0.3), self._d4_param, self._set_d4)
        self.register_constraint("raw_d4", drift_constraint)
        
        # D5
        self. register_parameter("raw_d5", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d5_prior", SmoothedBoxPrior(0.1, 0.3), self._d5_param, self._set_d5)
        self.register_constraint("raw_d5", drift_constraint)
        
        # D6
        self.register_parameter("raw_d6", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("d6_prior", SmoothedBoxPrior(0.3, 0.6), self._d6_param, self._set_d6)
        self.register_constraint("raw_d6", drift_constraint)
        
        print(f"[DEBUG] AresPriorMean initialized with 6 trainable drift lengths")

    # Getter/setter methods for each drift (FODO pattern)
    def _d1_param(self, m): return m.raw_d1_constraint.transform(m.raw_d1)
    def _set_d1(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d1)
        m.initialize(raw_d1=m.raw_d1_constraint.inverse_transform(v))
    
    def _d2_param(self, m): return m.raw_d2_constraint.transform(m.raw_d2)
    def _set_d2(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d2)
        m.initialize(raw_d2=m.raw_d2_constraint.inverse_transform(v))
    
    def _d3_param(self, m): return m.raw_d3_constraint.transform(m.raw_d3)
    def _set_d3(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d3)
        m.initialize(raw_d3=m.raw_d3_constraint.inverse_transform(v))
    
    def _d4_param(self, m): return m.raw_d4_constraint.transform(m.raw_d4)
    def _set_d4(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d4)
        m.initialize(raw_d4=m.raw_d4_constraint.inverse_transform(v))
    
    def _d5_param(self, m): return m.raw_d5_constraint.transform(m.raw_d5)
    def _set_d5(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d5)
        m.initialize(raw_d5=m.raw_d5_constraint.inverse_transform(v))
    
    def _d6_param(self, m): return m.raw_d6_constraint.transform(m.raw_d6)
    def _set_d6(self, m, v):
        if not torch.is_tensor(v): v = torch.as_tensor(v).to(m.raw_d6)
        m.initialize(raw_d6=m.raw_d6_constraint.inverse_transform(v))
    
    @property
    def d1(self): return self._d1_param(self)
    @d1.setter
    def d1(self, v): self._set_d1(self, v)
    
    @property
    def d2(self): return self._d2_param(self)
    @d2.setter
    def d2(self, v): self._set_d2(self, v)
    
    @property
    def d3(self): return self._d3_param(self)
    @d3.setter
    def d3(self, v): self._set_d3(self, v)
    
    @property
    def d4(self): return self._d4_param(self)
    @d4.setter
    def d4(self, v): self._set_d4(self, v)
    
    @property
    def d5(self): return self._d5_param(self)
    @d5.setter
    def d5(self, v): self._set_d5(self, v)
    
    @property
    def d6(self): return self._d6_param(self)
    @d6.setter
    def d6(self, v): self._set_d6(self, v)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.CV.angle = X[..., 2]
        self.Q3.k1 = X[..., 3]
        self.CH.angle = X[..., 4]
        
        # Update all drift lengths
        self.D1.length = self.d1
        self.D2.length = self.d2
        self.D3.length = self.d3
        self.D4.length = self.d4
        self.D5.length = self.d5
        self.D6.length = self.d6
        
        out_beam = self.segment(self.incoming_beam)
        beam_quality_mae = out_beam.mu_x.abs() + out_beam.mu_y.abs() + 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
        
        return beam_quality_mae

    def get_drift_values(self):
        """Extract all drift lengths"""
        return {
            "d1": float(self.d1.item()) if torch.is_tensor(self.d1) else float(self.d1),
            "d2": float(self.d2.item()) if torch.is_tensor(self.d2) else float(self.d2),
            "d3": float(self.d3.item()) if torch.is_tensor(self.d3) else float(self.d3),
            "d4": float(self.d4.item()) if torch.is_tensor(self.d4) else float(self.d4),
            "d5": float(self.d5.item()) if torch.is_tensor(self.d5) else float(self.d5),
            "d6": float(self.d6.item()) if torch.is_tensor(self.d6) else float(self.d6),
        }