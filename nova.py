"""Free-free emission from a classical nova shell at radio wavelengths
Following Hjellming et al. (1979), https://ui.adsabs.harvard.edu/abs/1979AJ.....84.1619H/abstract
And Osterbrock (1989): https://ui.adsabs.harvard.edu/abs/1989agna.book.....O/abstract

Free-free emission optical depth verified with:
https://www.cv.nrao.edu/~sransom/web/Ch4.html

Assumes Hubble-like expansion

Example:

>>> from nova import NovaShell
>>> n = NovaShell(
        Te=1e4 * u.K,
        M=8.6e-5 * u.Msun,
        d=800 * u.pc,
        v2=450 * u.km / u.s,
        v1=0.44 * 450 * u.km / u.s,
    )
>>> print(n.Snu(1e9*u.Hz, np.array([1000,2000])*u.d))
"""
from astropy import units as u, constants as c
import numpy as np
from astropy.modeling import models

nu0 = c.c * c.Ryd
# constant out in front of emissivity
# Osterbrock (1989) Eqn. 4.22
J0 = (
    (1 / (np.pi * 4))
    * (32 * 1**2 * c.e.gauss**4 * c.h / (3 * c.m_e**2 * c.c**3))
    * (np.pi * c.h * nu0 / (3 * c.k_B)) ** 0.5
).cgs


class NovaShell:
    @u.quantity_input(
        Te=u.K,
        M=u.Msun,
        d=u.pc,
        v1=u.km / u.s,
        v2=u.km / u.s,
        ne_by_rho=u.g**-1,
        r10=u.AU,
        r20=u.AU,
    )
    def __init__(
        self,
        Te,
        M,
        d,
        v1,
        v2,
        ne_by_rho=1 / c.m_p.cgs,
        r10=0 * u.AU,
        r20=0 * u.AU,
    ):
        """Classical nova shell with Hubble-type expansion
        Following Hjellming et al. (1979)

        :param Te: electron temperature
        :type Te: u.K
        :param M: shell total mass
        :type M: u.Msun
        :param d: distance
        :type d: u.pc
        :param v1: inner velocity
        :type v1: u.km/u.s
        :param v2: outer velocity
        :type v2: u.km/u.s
        :param ne_by_rho: electron density divided by mass density, defaults to 1/c.m_p.cgs
        :type ne_by_rho: 1/u.g, optional
        :param r10: inner radius at t=0, defaults to 0
        :type r10: u.AU, optional
        :param r20: outer radius at t=0, defaults to 0
        :type r20: u.AU, optional
        """
        self.Te = Te
        self.M = M
        self.v1 = v1
        self.v2 = v2
        self.d = d
        self.ne_by_rho = ne_by_rho
        self.r10 = r10
        self.r20 = r20

    @u.quantity_input
    def gaunt_ff(self, nu: u.Hz) -> u.dimensionless_unscaled:
        """Gaunt factor for free-free emission

        Hjellming et al. (1979), Eqn. 14

        :param nu: frequency
        :type nu: u.Hz
        :return: gaunt factor
        :rtype: u.dimensionless_unscaled
        """

        if np.any(nu > 1e12 * u.Hz):
            raise ValueError("Free-free gaunt factor is only valid for nu<1e12 Hz")

        return (3**0.5 / np.pi) * (
            17.7 + np.log(self.Te.value**1.5 / nu.to_value(u.Hz))
        )

    @u.quantity_input
    def jnu_ff(self, nu: u.Hz, ne: u.cm**-3) -> u.erg / u.s / u.cm**3 / u.Hz:
        """Volume emissivity for free-free emission

        Hjellming et al. (1979), Eqn. 13
        Osterbrock (1989), Eqn. 4.22

        :param nu: frequency
        :type nu: u.Hz
        :param ne: electron number density
        :type ne: u.cm**-3
        :return: volume emissivity
        :rtype: u.erg / u.s / u.cm**3 / u.Hz
        """
        return (
            J0
            * ne**2
            * self.Te**-0.5
            * self.gaunt_ff(nu)
            * np.exp(-(c.h * nu / c.k_B / self.Te).decompose())
        )

    @u.quantity_input
    def Bnu(self, nu: u.Hz) -> u.erg / u.s / u.cm**2 / u.Hz:
        """Blackbody intensity per steradian

        :param nu: frequency
        :type nu: u.Hz
        :return: specific intensity
        :rtype: u.erg/u.s/u.cm**2/u.Hz
        """
        return models.BlackBody(self.Te)(nu).to(
            u.erg / u.s / u.cm**2 / u.Hz, equivalencies=u.dimensionless_angles()
        )

    @u.quantity_input
    def alphanu_ff(self, nu: u.Hz, ne: u.cm**-3) -> u.cm**-1:
        """free-free apsorption coefficient

        :param nu: frequency
        :type nu: u.Hz
        :param ne: electron density
        :type ne: u.cm**-3
        :return: absorption
        :rtype: u.cm**-1
        """
        return self.jnu_ff(nu, ne) / self.Bnu(nu)

    @u.quantity_input
    def kappanu_ff(
        self,
        nu: u.Hz,
        ne: u.cm**-3,
        rho: u.g * u.cm**-3,
    ) -> u.cm**2 * u.g**-1:
        """Opacity for free-free

        :param nu: frequency
        :type nu: u.Hz
        :param ne: electron density
        :type ne: u.cm**-3
        :param rho: mass density
        :type rho: u.g/u.cm**3
        :return: opacity
        :rtype: u.cm**2/u.g
        """
        return self.alphanu_ff(nu, ne) / rho

    @u.quantity_input
    def epsilonnu_ff(
        self,
        nu: u.Hz,
        ne: u.cm**-3,
        rho: u.g * u.cm**-3,
    ) -> u.erg / u.s / u.Hz / u.g:
        """Mass emissivity for free-free

        :param nu: frequency
        :type nu: u.Hz
        :param ne: electron density
        :type ne: u.cm**-3
        :param rho: mass density
        :type rho: u.g/u.cm**3
        :return: emissivity
        :rtype: u.erg/u.s/u.Hz/u.g
        """
        return 4 * np.pi * self.jnu_ff(nu, ne) / rho

    @u.quantity_input
    def F(
        self,
        nu: u.Hz,
    ) -> u.erg * u.cm**3:
        """Convenience function for free-free calculation

        Ignoring the bound-free portion

        Hjellming et al. (1979), Eqn. 19

        :param nu: frequency
        :type nu: u.Hz
        :return: F value
        :rtype: u.erg*u.cm**3
        """

        return (
            (self.ne_by_rho) ** 2
            * np.exp(-(c.h * nu / c.k_B / self.Te).decompose())
            * (self.M / 4 / np.pi) ** 2
            * J0
            * self.Te**-0.5
            * self.gaunt_ff(nu)
        )

    @u.quantity_input
    def G(self, r: u.AU, a: u.AU) -> u.AU**-3:
        """
        Convenience function for free-free calculation

        Ignoring the bound-free portion

        Hjellming et al. (1979), Eqn. 19

        :param r: radius
        :type r: u.AU
        :param a: impact parameter
        :type a: u.AU
        :return: G value
        :rtype: u.AU**-3
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1 / a**2) * (
                (r**2 - a**2) ** 0.5 / r**2
                + (1 / a)
                * np.arccos((a / r).decompose()).to(
                    u.dimensionless_unscaled, equivalencies=u.dimensionless_angles()
                )
            )

    @u.quantity_input
    def r1(self, t: u.d) -> u.AU:
        """Inner radius of nova shell

        Hjellming et al. (1979): Eqn. 7

        :param t: time since ejection
        :type t: u.Quantity[u.d]
        :return: radius
        :rtype: u.AU
        """

        return self.r10 + self.v1 * t

    @u.quantity_input
    def r2(self, t: u.d) -> u.AU:
        """Outer radius of nova shell

        Hjellming et al. (1979): Eqn. 8

        :param t: time since ejection
        :type t: u.Quantity[u.d]
        :return: radius
        :rtype: u.AU
        """
        return self.r20 + self.v2 * t

    @u.quantity_input
    def rho(
        self,
        r: u.AU,
        t: u.d,
    ) -> u.g / u.cm**3:
        """Mass density of the shell

        Hjellming et al. (1979): Eqn. 11

        :param r: radius within the shell
        :type r: u.Quantity[u.AU]
        :param t: time since ejection
        :type t: u.Quantity[u.d]
        :return: density
        :rtype: u.g/u.cm**3
        """
        value = (self.M / (4 * np.pi * r**2)) / (self.r2(t) - self.r1(t))
        value[r < self.r1(t)] = 0
        value[r > self.r2(t)] = 0
        return value

    @u.quantity_input
    def tau(self, nu: u.Hz, t: u.d, a: u.AU) -> u.dimensionless_unscaled:
        """Nova shell optical depth

        Hjellming et al. (1979), Eqn. 20, 21

        :param nu: frequency
        :type nu: u.Hz
        :param t: time since ejection
        :type t: u.d
        :return: optical depth
        :rtype: u.dimensionless_unscaled
        """
        tau = np.zeros(len(a)) * u.dimensionless_unscaled
        with np.errstate(invalid="ignore"):
            # for 0 < a < r1
            tau1 = (
                (self.F(nu) / self.Bnu(nu))
                * (self.G(self.r2(t), a) - self.G(self.r1(t), a))
                / (self.r2(t) - self.r1(t)) ** 2
            )
            # for r1 < a < r2
            tau2 = (
                (self.F(nu) / self.Bnu(nu))
                * (self.G(self.r2(t), a))
                / (self.r2(t) - self.r1(t)) ** 2
            )
            # for a == 0
            tau3 = (
                (self.F(nu) / self.Bnu(nu))
                * (2 / 3)
                * (self.r1(t) ** -3 - self.r2(t) ** -3)
                / (self.r2(t) - self.r1(t)) ** 2
            )
        tau = tau1
        tau[a >= self.r1(t)] = tau2[a >= self.r1(t)]
        tau[a == 0] = tau3

        return tau

    @u.quantity_input
    def Snu(self, nu: u.Hz, t: u.d) -> u.mJy:
        """Flux density

        Hjellming et al. (1979), Eqn. 17

        :param nu: frequency
        :type nu: u.Hz
        :param t: time since ejection
        :type t: u.d
        :return: flux density
        :rtype: u.mJy
        """
        a = np.linspace(0, self.r2(t), 200)
        tau = self.tau(nu, t, a)
        return (
            2
            * np.pi
            * self.Bnu(nu)
            * np.trapz(a * (1 - np.exp(-tau)), a, axis=0)
            / self.d**2
        )

    @u.quantity_input
    def Snu_thin(self, nu: u.Hz, t: u.d) -> u.mJy:
        """Optically thin flux density, mostly useful as a check on the full calculation

        Hjellming et al. (1979), Eqn. 24


        :param nu: frequency
        :type nu: u.Hz
        :param t: time since ejection
        :type t: u.d
        :return: flux density
        :rtype: u.mJy
        """
        return (
            4
            * np.pi
            * self.F(nu)
            / (self.d**2 * (self.r2(t) - self.r1(t)) * self.r1(t) * self.r2(t))
        )


# n = NovaShell(
#     Te=1e4 * u.K,
#     M=8.6e-5 * u.Msun,
#     d=800 * u.pc,
#     v2=450 * u.km / u.s,
#     v1=0.44 * 450 * u.km / u.s,
# )

# # # t = np.logspace(2, 3.7) * u.d
# # t = np.array([100, 1000, 5000]) * u.d
# # a = np.linspace(0, n.r2(t), 200)
# # nu = 3 * u.GHz
# # tau = n.tau(nu, t, a)
# # Snu = n.Snu(nu, t)

# nu = 7 * u.GHz
# t = 1000 * u.d
# r = np.linspace(0, n.r2(t), 200)
# rho = n.rho(r, t)
# ne = rho / c.m_p
# EM = 2 * np.trapz(ne**2, r.to(u.pc)).to(u.cm**-6 * u.pc)
# gff = np.log(4.955e-2 * nu.to_value(u.GHz) ** -1) + 1.5 * np.log(n.Te.value)
# gff2 = n.gaunt_ff(nu)
# tau2 = 3.01e-2 * (n.Te.value) ** -1.5 * nu.to_value(u.GHz) ** -2 * EM.value * gff
# tau = n.tau(nu, t, np.array([0]) * u.AU)
