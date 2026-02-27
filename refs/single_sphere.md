# Dielectric sphere in a uniform \(\mathbf{E}_0 = E_0\,\hat{\mathbf z}\): equivalent potential (outside)

Assume a dielectric sphere of radius \(a\) with relative permittivity \(\varepsilon_r\) embedded in vacuum (outside permittivity \(\varepsilon_0\)). A uniform external field \(\mathbf E_0 = E_0\,\hat{\mathbf z}\) is applied.

## Equivalent (external) potential for \(r \ge a\)

The field outside the sphere is equivalent to the superposition of:
1) the original uniform-field potential, and  
2) a point dipole at the center aligned with \(+z\).

Using spherical coordinates \((r,\theta)\) (\(\theta\) is the angle from \(+z\)):

\[
\Phi_{\text{eq}}(r,\theta)= -E_0\, r\cos\theta \, +\, \frac{1}{4\pi\varepsilon_0}\,\frac{p\cos\theta}{r^{2}} .
\]

The induced dipole moment is:
\[
p = 4\pi\varepsilon_0 a^{3}\, \frac{\varepsilon_r-1}{\varepsilon_r+2}\, E_0 .
\]

## Perturbation potential (relative to the applied uniform field)

\[
\Delta\Phi(r,\theta)=\Phi_{\text{eq}} - (-E_0 r\cos\theta)
= \frac{1}{4\pi\varepsilon_0}\,\frac{p\cos\theta}{r^{2}}
= a^3\frac{\varepsilon_r-1}{\varepsilon_r+2}\,E_0\,\frac{\cos\theta}{r^{2}} .
\]

## Note (non-vacuum background)

If the surrounding medium has permittivity \(\varepsilon_{\text{out}}\neq \varepsilon_0\), replace \(\varepsilon_0\to\varepsilon_{\text{out}}\) and interpret
\(\varepsilon_r = \varepsilon_{\text{in}}/\varepsilon_{\text{out}}\).
