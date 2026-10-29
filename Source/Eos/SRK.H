#ifndef SRK_H
#define SRK_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include "mechanism.H"
#include "PhysicsConstants.H"

namespace pele::physics::eos {

struct SRK
{
  using eos_type = SRK;

  static std::string identifier() { return "SRK"; }

  // ReallArrayLike can be anything with an [] operator that returns
  // amrex::Real&
  template <typename RealArrayLike>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
  RY2R(const RealArrayLike& RY, amrex::Real& R, int first = 0)
  {
    R = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      R += RY[first + i];
    }
  }

  // ReallArrayLike can be anything with an [] operator that returns
  // amrex::Real&
  template <typename RealArrayLike1, typename RealArrayLike2>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void RY2RRinvY(
    const RealArrayLike1& RY,
    amrex::Real& R,
    amrex::Real& Rinv,
    RealArrayLike2&& Y)
  {
    RY2R(RY, R);
    Rinv = 1.0 / R;
    for (int i = 0; i < NUM_SPECIES; i++) {
      Y[i] = RY[i] * Rinv;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void RY2dRdY(
    const amrex::Real& /*R*/,
    const amrex::Real* /*Y[]*/,
    amrex::Real* /*dRdY[]*/)
  {
    amrex::Error("RY2dRdY not applicable for SRK EOS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void MixingRuleAmBm(
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& am,
    amrex::Real& bm)
  {
    am = 0.0;
    bm = 0.0;
    AMREX_ASSERT(T > 0.0);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];

    // Combine as follows: add diagonal to am in first loop, loop upper triang
    // in second loop
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
      bm += Y[ii] * Bi[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        am += Y[ii] * Y[jj] * amloc[ii] * amloc[jj];
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void MixingRuleAm(
    const amrex::Real T, const amrex::Real Y[NUM_SPECIES], amrex::Real& am)
  {
    am = 0.0;
    AMREX_ASSERT(T > 0.0);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];

    // Combine as follows: add diagonal to am in first loop, loop upper triang
    // in second loop
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        am += Y[ii] * Y[jj] * amloc[ii] * amloc[jj];
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void MixingRuleBm(const amrex::Real Y[NUM_SPECIES], amrex::Real& bm)
  {
    bm = 0.0;
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      bm += Y[ii] * Bi[ii];
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_dAmdT(
    const amrex::Real T, const amrex::Real Y[NUM_SPECIES], amrex::Real& dAmdT)
  {
    dAmdT = 0.0;
    AMREX_ASSERT(T > 0.0);
    const amrex::Real oneOverT = 1.0 / T;
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real amlocder[NUM_SPECIES];

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
      amlocder[ii] =
        -0.5 * Fomega[ii] * sqrtAsti[ii] * oneOverT * sqrtT * sqrtOneOverTc[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dAmdT +=
          Y[ii] * Y[jj] * (amloc[ii] * amlocder[jj] + amloc[jj] * amlocder[ii]);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_dAmdY(
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real dAmdY[NUM_SPECIES])
  {
    AMREX_ASSERT(T > 0.0);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      dAmdY[ii] = 0.0;
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dAmdY[ii] += 2.0 * Y[jj] * (amloc[ii] * amloc[jj]);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_d2AmdY2(
    const amrex::Real T,
    const amrex::Real* /*Y[]*/,
    amrex::Real d2AmdY2[NUM_SPECIES][NUM_SPECIES])
  {
    AMREX_ASSERT(T > 0.0);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        d2AmdY2[ii][jj] = 2.0 * amloc[ii] * amloc[jj];
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_d2AmdTY(
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real d2AmdTY[NUM_SPECIES])
  {
    AMREX_ASSERT(T > 0.0);
    const amrex::Real oneOverT = 1.0 / T;
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real amlocder[NUM_SPECIES];

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      d2AmdTY[ii] = 0.0;
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
      amlocder[ii] =
        -0.5 * Fomega[ii] * sqrtAsti[ii] * oneOverT * sqrtT * sqrtOneOverTc[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        d2AmdTY[ii] +=
          2.0 * Y[jj] * (amloc[ii] * amlocder[jj] + amloc[jj] * amlocder[ii]);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void Calc_CompressFactor_Z(
    amrex::Real& Z,
    const amrex::Real am,
    const amrex::Real bm,
    const amrex::Real P,
    const amrex::Real T,
    const amrex::Real Wbar)
  {
    amrex::Real RmT = Constants::RU / Wbar * T;
    amrex::Real B1 = bm * P / RmT;
    amrex::Real R1 = RmT;
    amrex::Real R2 = R1 * RmT;
    amrex::Real R3 = R2 * RmT;
    amrex::Real alpha = -1.0;
    amrex::Real beta = (am * P - bm * P * bm * P) / R2 - B1;
    amrex::Real gamma = -(am * bm * P * P) / R3;
    amrex::Real Q = (alpha * alpha - 3.0 * beta) / 9.0;
    amrex::Real R =
      (2.0 * alpha * alpha * alpha - 9.0 * alpha * beta + 27.0 * gamma) / 54.0;

    // Multiple roots of cubic
    const amrex::Real third = 1.0 / 3.0;
    if ((Q * Q * Q - R * R) > 0) {
      const amrex::Real sqrtQ = std::sqrt(Q);
      const amrex::Real theta = std::acos(R / (Q * sqrtQ));
      const amrex::Real Z1 =
        -2.0 * sqrtQ * std::cos(theta * third) - alpha * third;
      const amrex::Real Z2 =
        -2.0 * sqrtQ * std::cos((theta + 2.0 * M_PI) * third) - alpha * third;
      const amrex::Real Z3 =
        -2.0 * sqrtQ * std::cos((theta + 4.0 * M_PI) * third) - alpha * third;
      Z = amrex::max<amrex::Real>(Z1, Z2);
      Z = amrex::max<amrex::Real>(Z, Z3);
    } else {
      Z = -std::copysign(1.0, R) *
            (std::pow((std::sqrt(R * R - Q * Q * Q) + std::abs(R)), third) +
             Q / (pow(std::sqrt(R * R - Q * Q * Q) + std::abs(R), third))) -
          alpha * third;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_d2AmdT2(
    const amrex::Real T, const amrex::Real Y[NUM_SPECIES], amrex::Real& d2AmdT2)
  {
    AMREX_ASSERT(T > 0.0);
    const amrex::Real oneOverT = 1.0 / T;
    const amrex::Real tmp1 = -0.5 * oneOverT;
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real amlocder[NUM_SPECIES];

    d2AmdT2 = 0.0;

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
      amlocder[ii] =
        -0.5 * Fomega[ii] * sqrtAsti[ii] * oneOverT * sqrtT * sqrtOneOverTc[ii];
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        d2AmdT2 += tmp1 * Y[ii] * Y[jj] *
                   (-4.0 * T * amlocder[ii] * amlocder[jj] +
                    amloc[ii] * amlocder[jj] + amloc[jj] * amlocder[ii]);
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void Calc_Am_and_derivs(
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& am,
    amrex::Real& dAmdT,
    amrex::Real& d2AmdT2)
  {
    AMREX_ASSERT(T > 0.0);
    const amrex::Real oneOverT = 1.0 / T;
    const amrex::Real tmp1 = -0.5 * oneOverT;
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real amlocder[NUM_SPECIES];

    // Compute species-dependent intermediates
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amloc[ii] =
        (1.0 + Fomega[ii] * (1.0 - sqrtT * sqrtOneOverTc[ii])) * sqrtAsti[ii];
      amlocder[ii] =
        Fomega[ii] * sqrtAsti[ii] * sqrtOneOverTc[ii]; // *-0.5*oneOverT*sqrtT
    }

    // initialize sums to 0
    am = 0.0;
    dAmdT = 0.0;
    d2AmdT2 = 0.0;

    // compute off-diagonal elements of sum using symmetry
#if NUM_SPECIES > 1
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      for (int jj = ii + 1; jj < NUM_SPECIES; jj++) {
        am += Y[ii] * Y[jj] * amloc[ii] * amloc[jj];
        dAmdT +=
          Y[ii] * Y[jj] * (amloc[ii] * amlocder[jj] + amloc[jj] * amlocder[ii]);
        d2AmdT2 += Y[ii] * Y[jj] * amlocder[ii] * amlocder[jj];
      }
    }
    am *= 2.0;
    dAmdT *= 2.0;
    d2AmdT2 *= 2.0;
#endif

    // Compute on diagonal elements of sum
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      am += Y[ii] * Y[ii] * amloc[ii] * amloc[ii];
      dAmdT += 2.0 * Y[ii] * Y[ii] * (amloc[ii] * amlocder[ii]);
      d2AmdT2 += Y[ii] * Y[ii] * amlocder[ii] * amlocder[ii];
    }

    // factor in constants
    dAmdT *= tmp1 * sqrtT;
    d2AmdT2 = -d2AmdT2;
    d2AmdT2 += dAmdT;
    d2AmdT2 *= tmp1;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void molecular_weight(amrex::Real mw[NUM_SPECIES]) { get_mw(mw); }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void inv_molecular_weight(amrex::Real imw[NUM_SPECIES])
  {
    get_imw(imw);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void TY2Cp(
    const amrex::Real /*T*/, const amrex::Real* /*Y[]*/, amrex::Real& /*Cp*/)
  {
    amrex::Error("TY2Cp not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2Cp(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& Cp)
  {
    amrex::Real bm, am, dAmdT, d2AmdT2;
    amrex::Real K1, Cpig = 0.0, wbar = 0.0, tau, Rm;
    amrex::Real dpdT, dpdtau, dhmdT, dhmdtau;

    CKMMWY(Y, wbar);
    MixingRuleBm(Y, bm);
    Calc_Am_and_derivs(T, Y, am, dAmdT, d2AmdT2);

    tau = 1.0 / R;
    K1 = (1.0 / bm) * log1p(bm * R);
    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    Rm = Constants::RU / wbar;

    dpdT = Rm * InvEosT1Denom - dAmdT * InvEosT2Denom;
    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
             am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;

    CKCPBS(T, Y, Cpig);
    dhmdT =
      Cpig + T * d2AmdT2 * K1 - dAmdT * InvEosT3Denom + Rm * bm * InvEosT1Denom;
    dhmdtau = -(T * dAmdT - am) * InvEosT2Denom +
              am * InvEosT3Denom * InvEosT3Denom -
              Rm * T * bm * InvEosT1Denom * InvEosT1Denom;
    Cp = dhmdT - (dhmdtau / dpdtau) * dpdT;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void TY2Cv(
    const amrex::Real /*T*/, const amrex::Real* /*Y[]*/, amrex::Real& /*Cp*/)
  {
    amrex::Error("TY2Cv not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2Cv(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& Cv)
  {
    amrex::Real am, bm, d2AmdT2;
    // Alternative: MixingRuleBm then Calc_Am_and_derivs
    MixingRuleAmBm(T, Y, am, bm);
    Calc_d2AmdT2(T, Y, d2AmdT2);
    CKCVBS(T, Y, Cv);
    Cv += T * d2AmdT2 * (1.0 / bm) * log1p(bm * R);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void T2Cpi(const amrex::Real /*T*/, amrex::Real* /*Cpi[]*/)
  {
    amrex::Error("T2Cpi not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void
  EY2T(const amrex::Real /*E*/, const amrex::Real* /*Y[]*/, amrex::Real& /*T*/)
  {
    // For Fuego this function is really just a wrapper for GET_T_GIVEN_EY
    // In SRK this will be different probably
    amrex::Error("EY2T not physically possible for this EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void REY2T(
    const amrex::Real R,
    const amrex::Real E,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& T)
  {
    // NOTE: for this function T is the output, but the input T serves as the
    // initial guess for Newton iteration, so it must be initialized to
    // a reasonable initial guess.
    amrex::Real Tn;
    amrex::Real bm, am, dAmdT, d2AmdT2;
    amrex::Real K1, Cv = 0.0;
    int nIter = 0;
    amrex::Real fzero = 1.0;
    amrex::Real Eig = 0.0;
    MixingRuleBm(Y, bm);
    K1 = (1.0 / bm) * log1p(bm * R);

    // Use input T as initial guess
    Tn = T;
    AMREX_ASSERT(Tn > 0.0);

    amrex::Real dT = 100000.0;
    while (std::abs(dT) > convCritT && nIter < maxIter) {
      nIter++;
      Calc_Am_and_derivs(Tn, Y, am, dAmdT, d2AmdT2);
      // ideal gas internal energy
      CKUBMS(Tn, Y, Eig);
      // ideal gas heat capacity
      CKCVBS(Tn, Y, Cv);
      // real gas heat capacity
      Cv += Tn * d2AmdT2 * K1;

      // Take difference between E(Tn) and E and iterate
      fzero = -E + Eig + (Tn * dAmdT - am) * K1;
      dT = fzero / Cv;
      Tn -= dT;
    }
    T = Tn;
    AMREX_ASSERT(nIter < maxIter);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void
  HY2T(const amrex::Real /*H*/, const amrex::Real* /*Y[]*/, amrex::Real& /*T*/)
  {
    // In SRK this  function is not possible
    amrex::Error("HY2T not physically possible for this EoS");
  }

  // RHY2T is implemented below, next to the other Newton inversions.

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2P(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& P)
  {
    amrex::Real wbar = 0.0, tau, am, bm;
    tau = 1.0 / R;
    CKMMWY(Y, wbar);
    MixingRuleAmBm(T, Y, am, bm);
    P = (Constants::RU / wbar) * T / (tau - bm) - am / (tau * (tau + bm));
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RYET2P(
    const amrex::Real R,
    const amrex::Real Y[NUM_SPECIES],
    const amrex::Real E,
    amrex::Real& T,
    amrex::Real& P)
  {
    // Note: this function was written to have the same behavior as the Fuego
    // version (T gets overwritten, so the function is actually RYE2TP)
    // I'm not sure if this behavior makes sense. The function is only called
    // once in PeleC, in pc_derpmmserror. One MixingRuleAm could be eliminated
    // by expanding these function explicitly, but that's small relative to
    // the overall cost. By using T as in input (unlike Fuego) iteration would
    // be eliminated and the overall cost would be greatly reduced. - BAP
    REY2T(R, E, Y, T);
    RTY2P(R, T, Y, P);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RYP2T(
    const amrex::Real R,
    const amrex::Real Y[NUM_SPECIES],
    const amrex::Real P,
    amrex::Real& T)
  {
    amrex::Real am, bm, dAmdT;
    amrex::Real tau, wbar = 0.0, Rm, Pnp1;
    amrex::Real eosT1Denom, eosT2Denom, InvEosT1Denom, InvEosT2Denom;
    int nIter;

    // Precalculate some variables
    tau = 1.0 / R;
    CKMMWY(Y, wbar);
    Rm = Constants::RU / wbar;

    // Use ideal gas to get an initial guess
    T = P * tau / Rm;
    MixingRuleAmBm(T, Y, am, bm);

    eosT1Denom = tau - bm;
    eosT2Denom = tau * (tau + bm);
    InvEosT1Denom = 1.0 / eosT1Denom;
    InvEosT2Denom = 1.0 / eosT2Denom;

    Pnp1 = Rm * T * InvEosT1Denom - am * InvEosT2Denom;
    nIter = 0;

    // Newton Iteration loop
    while (std::abs(P - Pnp1) > convCritP && nIter < maxIter) {
      nIter += 1;
      Calc_dAmdT(T, Y, dAmdT);
      amrex::Real dpdT = Rm * InvEosT1Denom - dAmdT * InvEosT2Denom;
      // Newton iteration: Tnext = T - (p(T,Y,R) - Ptarget)/dpdT
      T -= (Pnp1 - P) / dpdT;
      // Calculate updated pressure to use as a convergence criterion
      MixingRuleAm(T, Y, am);
      Pnp1 = Rm * T * InvEosT1Denom - am * InvEosT2Denom;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2C(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real acti[NUM_SPECIES])
  {
    amrex::Real am, bm, dAmdY[NUM_SPECIES];
    amrex::Real K1, InvEosT1Denom, InvEosT3Denom, wbar = 0.0, RmT, InvRuT,
                                                  InvBm, tau;

    MixingRuleAmBm(T, Y, am, bm);
    Calc_dAmdY(T, Y, dAmdY);
    CKMMWY(Y, wbar);

    InvBm = 1.0 / bm;
    tau = 1.0 / R;
    K1 = InvBm * log1p(bm * R);
    RmT = Constants::RU / wbar * T;
    InvRuT = 1.0 / (Constants::RU * T);
    InvEosT1Denom = 1.0 / (tau - bm);
    InvEosT3Denom = 1.0 / (tau + bm);

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      acti[ii] =
        (exp(
           (mw(ii) * InvRuT) *
           (RmT * InvEosT1Denom * Bi[ii] - K1 * dAmdY[ii] +
            am * InvBm * K1 * Bi[ii] - am * InvEosT3Denom * Bi[ii] * InvBm)) *
         Y[ii] * imw(ii) * InvEosT1Denom);
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2WDOT(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real WDOT[NUM_SPECIES])
  {
    amrex::Real C[NUM_SPECIES];
    RTY2C(R, T, Y, C);
    CKWC(T, C, WDOT);

    for (int n = 0; n < NUM_SPECIES; n++) {
      WDOT[n] *= mw(n);
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void RTY2JAC(
    const amrex::Real /*R*/,
    const amrex::Real /*T*/,
    const amrex::Real* /*Y[]*/,
    amrex::Real* /*JAC[]*/,
    const int /*HP*/)
  {
    amrex::Error("RTY2JAC not supported with SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void T2Ei(const amrex::Real /*T*/, amrex::Real* /*Ei[]*/)
  {
    amrex::Error("T2Ei not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2Ei(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real Ei[NUM_SPECIES])
  {
    amrex::Real am, bm, dAmdT, dAmdY[NUM_SPECIES], d2AmdTY[NUM_SPECIES];
    amrex::Real K1, inv_bm, InvEosT3Denom;

    // ideal gas portion
    CKUMS(T, Ei);

    // non-ideal portion : Combine Am and Derivs for Speedup?
    MixingRuleAmBm(T, Y, am, bm);
    Calc_dAmdT(T, Y, dAmdT);
    Calc_dAmdY(T, Y, dAmdY);
    Calc_d2AmdTY(T, Y, d2AmdTY);

    inv_bm = 1.0 / bm;
    K1 = inv_bm * log1p(bm * R);
    InvEosT3Denom = 1.0 / (1.0 / R + bm);

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      Ei[ii] += (T * d2AmdTY[ii] - dAmdY[ii]) * K1;
      Ei[ii] += (T * dAmdT - am) * Bi[ii] * (InvEosT3Denom - K1) * inv_bm;
    }
  }

  // Function added for SRK
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2E(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& E)
  {
    amrex::Real Ei[NUM_SPECIES];
    amrex::Real am, bm, dAmdT, K1;

    // Calculate ideal gas portion
    E = 0.0;
    CKUMS(T, Ei);
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      E += Ei[ii] * Y[ii];
    }

    // Add in non-ideal portion
    MixingRuleAmBm(T, Y, am, bm);
    Calc_dAmdT(T, Y, dAmdT);
    // below was log(1 + bm/tau), tau = 1/R so this is simpler)
    K1 = (1.0 / bm) * log1p(bm * R);
    E += (T * dAmdT - am) * K1;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void Y2X(const amrex::Real Y[NUM_SPECIES], amrex::Real X[NUM_SPECIES])
  {
    CKYTX(Y, X);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void Y2Z(const amrex::Real Y[NUM_SPECIES], amrex::Real Z[NUM_ELEMENTS])
  {
    amrex::Real atm_wt[NUM_ELEMENTS];
    atomicWeight(atm_wt);
    amrex::Real inv_mw[NUM_SPECIES];
    get_imw(inv_mw);
    int ecomp[NUM_SPECIES * NUM_ELEMENTS];
    CKNCF(ecomp);

    for (int k = 0; k < NUM_ELEMENTS; k++) {
      Z[k] = 0.0;
    }
    for (int i = 0; i < NUM_SPECIES; i++) {
      for (int k = 0; k < NUM_ELEMENTS; k++) {
        Z[k] += ecomp[i * NUM_ELEMENTS + k] * atm_wt[k] * inv_mw[i] * Y[i];
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void X2Y(const amrex::Real X[NUM_SPECIES], amrex::Real Y[NUM_SPECIES])
  {
    CKXTY(X, Y);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void PYT2RE(
    const amrex::Real P,
    const amrex::Real Y[NUM_SPECIES],
    const amrex::Real T,
    amrex::Real& R,
    amrex::Real& E)
  {
    // Calculate RHO
    amrex::Real am, bm, Z, wbar = 0.0, K1;
    MixingRuleAmBm(T, Y, am, bm);
    CKMMWY(Y, wbar);
    Calc_CompressFactor_Z(Z, am, bm, P, T, wbar);
    R = P * wbar / (Z * Constants::RU * T);

    // Calculate E - ideal gas portion then add non-ideal portion
    amrex::Real Ei[NUM_SPECIES], dAmdT;
    E = 0.0;
    CKUMS(T, Ei);
    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      E += Ei[ii] * Y[ii];
    }
    Calc_dAmdT(T, Y, dAmdT);
    K1 = (1.0 / bm) * log1p(bm * R); // R = 1/tau
    E += (T * dAmdT - am) * K1;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void PYT2R(
    const amrex::Real P,
    const amrex::Real Y[NUM_SPECIES],
    const amrex::Real T,
    amrex::Real& R)
  {
    amrex::Real am, bm, Z, wbar = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    CKMMWY(Y, wbar);
    Calc_CompressFactor_Z(Z, am, bm, P, T, wbar);
    R = P * wbar / (Z * Constants::RU * T);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RYP2E(
    const amrex::Real R,
    const amrex::Real Y[NUM_SPECIES],
    const amrex::Real P,
    amrex::Real& E)
  {
    amrex::Real T;
    // Note: could get away with one fewer MixingRuleAm by writing this
    // out explicitly, but that is trivial relative to cost of iteration
    // in RYP2T.
    RYP2T(R, Y, P, T);
    RTY2E(R, T, Y, E);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void T2Hi(const amrex::Real /*T*/, amrex::Real* /*Hi[]*/)
  {
    amrex::Error("T2Hi not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2Hi(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real Hi[NUM_SPECIES])
  {
    amrex::Real am, bm, dAmdT, d2AmdT2;
    amrex::Real dAmdY[NUM_SPECIES], d2AmdTY[NUM_SPECIES];
    amrex::Real wbar = 0.0;
    amrex::Real dpdtau, dhmdtau;

    // Ideal gas part
    CKHMS(T, Hi);

    // Non-ideal part: Could be optimized a bit more by combining all derivative
    // calls
    MixingRuleBm(Y, bm);
    Calc_Am_and_derivs(T, Y, am, dAmdT, d2AmdT2);
    Calc_dAmdY(T, Y, dAmdY);
    Calc_d2AmdTY(T, Y, d2AmdTY);
    CKMMWY(Y, wbar);

    amrex::Real tau = 1.0 / R;
    amrex::Real K1 = (1.0 / bm) * log1p(bm * R);
    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    amrex::Real Rm = Constants::RU / wbar;

    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
             am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;
    dhmdtau = -(T * dAmdT - am) * InvEosT2Denom +
              am * InvEosT3Denom * InvEosT3Denom -
              Rm * T * bm * InvEosT1Denom * InvEosT1Denom;

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amrex::Real Rmk = Constants::RU * imw(ii);
      amrex::Real dpdYk = Rmk * T * InvEosT1Denom - dAmdY[ii] * InvEosT2Denom +
                          Bi[ii] * (Rm * T * InvEosT1Denom * InvEosT1Denom +
                                    am * InvEosT2Denom * InvEosT3Denom);
      amrex::Real dhmdYk =
        Hi[ii] + (T * d2AmdTY[ii] - dAmdY[ii]) * K1 -
        Bi[ii] * (T * dAmdT - am) * (K1 / bm - InvEosT3Denom / bm) +
        am * Bi[ii] * InvEosT3Denom * InvEosT3Denom -
        InvEosT3Denom * dAmdY[ii] + Rmk * T * bm * InvEosT1Denom +
        Rm * T * Bi[ii] * (InvEosT1Denom + bm * InvEosT1Denom * InvEosT1Denom);
      Hi[ii] = dhmdYk - (dhmdtau / dpdtau) * dpdYk;
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void S(amrex::Real& s) { s = 1.0; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void
  TY2G(const amrex::Real /*T*/, const amrex::Real* /*Y[]*/, amrex::Real& /*G*/)
  {
    amrex::Error("TY2G not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2G(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& G)
  {
    amrex::Real bm, am, dAmdT, d2AmdT2;
    amrex::Real K1, Cpig = 0.0, Cp, Cv = 0.0, wbar = 0.0, tau, Rm;
    amrex::Real P, dpdT, dpdtau, dhmdT, dhmdtau;

    CKMMWY(Y, wbar);
    MixingRuleBm(Y, bm);
    Calc_Am_and_derivs(T, Y, am, dAmdT, d2AmdT2);

    tau = 1.0 / R;
    K1 = (1.0 / bm) * log1p(bm * R);

    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    Rm = Constants::RU / wbar;

    dpdT = Rm * InvEosT1Denom - dAmdT * InvEosT2Denom;
    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
             am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;

    CKCPBS(T, Y, Cpig);
    dhmdT =
      Cpig + T * d2AmdT2 * K1 - dAmdT * InvEosT3Denom + Rm * bm * InvEosT1Denom;
    dhmdtau = -(T * dAmdT - am) * InvEosT2Denom +
              am * InvEosT3Denom * InvEosT3Denom -
              Rm * T * bm * InvEosT1Denom * InvEosT1Denom;
    Cp = dhmdT - (dhmdtau / dpdtau) * dpdT;

    CKCVBS(T, Y, Cv);
    Cv += T * d2AmdT2 * K1;

    P = (Constants::RU / wbar) * T * InvEosT1Denom - am * InvEosT2Denom;
    G = -tau * Cp * dpdtau / (P * Cv);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RPY2Cs(
    const amrex::Real R,
    const amrex::Real P,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& Cs)
  {
    amrex::Real T, G;
    // RTY2Cs involves some redundant Am calcs but add cost is small relative
    // to iteration for RYP2T
    RYP2T(R, Y, P, T);
    RTY2G(R, T, Y, G);
    Cs = std::sqrt(G * P / R);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2Cs(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& Cs)
  {
    amrex::Real P, G;
    // RTY2P involves one redundant MixingRuleAm
    RTY2G(R, T, Y, G);
    RTY2P(R, T, Y, P);
    Cs = std::sqrt(G * P / R);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void TY2H(
    const amrex::Real /*T*/,
    const amrex::Real* /*Y[NUM_SPECIES]*/,
    amrex::Real& /*H*/)
  {
    amrex::Error("TY2H not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2H(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& H)
  {
    amrex::Real E, P;
    RTY2E(R, T, Y, E);
    RTY2P(R, T, Y, P);
    H = E + P / R;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void Y2WBAR(const amrex::Real Y[NUM_SPECIES], amrex::Real& WBAR)
  {
    amrex::Real summ = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      summ += Y[i] * imw(i);
    }
    WBAR = 1.0 / summ;
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void RPE2dpdr_e(
    const amrex::Real /*R*/,
    const amrex::Real /*P*/,
    const amrex::Real /* E */,
    amrex::Real& /*dpdr_e*/)
  {
    amrex::Error("RPE2dpdr_e not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void RG2dpde(
    const amrex::Real /*R*/, const amrex::Real /*G*/, amrex::Real& /*dpde*/)
  {
    amrex::Error("RG2dpde not physically possible for SRK EoS");
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2dpde_dpdre(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& dpde,
    amrex::Real& dpdr_e)
  {
    amrex::Real am, bm, dAmdT, d2AmdT2, wbar = 0.0, Cv = 0.0;

    CKMMWY(Y, wbar);
    MixingRuleBm(Y, bm);
    Calc_Am_and_derivs(T, Y, am, dAmdT, d2AmdT2);

    amrex::Real Rm = Constants::RU / wbar;
    amrex::Real tau = 1.0 / R;
    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    CKCVBS(T, Y, Cv);
    Cv += T * d2AmdT2 * (1.0 / bm) * log1p(bm * R);

    amrex::Real dpdT = Rm * InvEosT1Denom - dAmdT * InvEosT2Denom;
    amrex::Real dedtau = InvEosT2Denom * (am - T * dAmdT);
    amrex::Real dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
                         am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;
    dpde = dpdT / Cv;
    dpdr_e = -tau * tau * (dpdtau - dedtau * dpde);
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2transport(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real diP[NUM_SPECIES],
    amrex::Real dijY[NUM_SPECIES][NUM_SPECIES])
  {
    amrex::Real bm, am, dAmdY[NUM_SPECIES], d2AmdY2[NUM_SPECIES][NUM_SPECIES];
    amrex::Real wbar = 0.0;
    amrex::Real dpdtau, dpdYk[NUM_SPECIES];

    // Optimize by combining Am derivate calls?
    CKMMWY(Y, wbar);
    MixingRuleAmBm(T, Y, am, bm);
    Calc_dAmdY(T, Y, dAmdY);
    Calc_d2AmdY2(T, Y, d2AmdY2);

    amrex::Real tau = 1.0 / R;
    amrex::Real InvBm = 1.0 / bm;
    amrex::Real K1 = (1.0 / bm) * log1p(bm * R);
    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    amrex::Real Rm = Constants::RU / wbar;

    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
             am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amrex::Real Rmk = Constants::RU * imw(ii);
      dpdYk[ii] = Rmk * T * InvEosT1Denom - dAmdY[ii] * InvEosT2Denom +
                  Bi[ii] * (Rm * T * InvEosT1Denom * InvEosT1Denom +
                            am * InvEosT2Denom * InvEosT3Denom);
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {

      diP[ii] = (Y[ii] * wbar / (Rm * T) *
                 ((dAmdY[ii] - am * Bi[ii] * InvBm) * InvEosT2Denom +
                  am * Bi[ii] * InvBm * InvEosT1Denom * InvEosT1Denom)) -
                Y[ii] * (Bi[ii] * InvEosT1Denom * InvEosT1Denom +
                         wbar * imw(ii) * InvEosT1Denom);
      diP[ii] /= dpdtau;

      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dijY[ii][jj] = 0.0;
      }
      dijY[ii][ii] = wbar * imw(ii);
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dijY[ii][jj] +=
          -diP[ii] * dpdYk[jj] +
          wbar * Y[ii] * InvEosT1Denom * (Bi[ii] * imw(jj) + Bi[jj] * imw(ii)) +
          Y[ii] * InvEosT1Denom * InvEosT1Denom * Bi[ii] * Bi[jj] +
          Y[ii] / (Rm * T) *
            ((K1 - InvEosT3Denom) * dAmdY[jj] * Bi[ii] * InvBm -
             K1 * d2AmdY2[ii][jj] +
             (K1 * dAmdY[ii] - InvEosT3Denom) * Bi[ii] * InvBm +
             ((-2.0 * am * K1 + am * InvEosT3Denom) * Bi[ii] * Bi[jj] * InvBm *
              InvBm) +
             ((am * InvEosT3Denom * InvBm +
               am * InvEosT3Denom * InvEosT3Denom) *
              Bi[ii] * Bi[jj] * InvBm));
      }
    }
  }

  // Same Helmholtz linearization as RTY2transport, with four residual-Jacobian
  // corrections relative to a^r = -Rm T ln(1-bm/tau) - am K1:
  //   1. diP attractive prefactor Y/(Rm T), not Y*Wbar/(Rm T)
  //   2. diP attractive last pole 1/(tau+bm)^2, not 1/(tau-bm)^2
  //   3. dijY covolume cross term sign is minus
  //   4. (K1-Inv3)*dAmdY[ii]*Bi[jj]/bm, not the j-independent
  //      (K1*dAmdY[ii]-Inv3)*Bi[ii]/bm
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void RTY2transport2(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real diP[NUM_SPECIES],
    amrex::Real dijY[NUM_SPECIES][NUM_SPECIES])
  {
    amrex::Real bm, am, dAmdY[NUM_SPECIES], d2AmdY2[NUM_SPECIES][NUM_SPECIES];
    amrex::Real wbar = 0.0;
    amrex::Real dpdtau, dpdYk[NUM_SPECIES];

    CKMMWY(Y, wbar);
    MixingRuleAmBm(T, Y, am, bm);
    Calc_dAmdY(T, Y, dAmdY);
    Calc_d2AmdY2(T, Y, d2AmdY2);

    amrex::Real tau = 1.0 / R;
    amrex::Real InvBm = 1.0 / bm;
    amrex::Real K1 = (1.0 / bm) * log1p(bm * R);
    amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    amrex::Real Rm = Constants::RU / wbar;

    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom +
             am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;

    for (int ii = 0; ii < NUM_SPECIES; ii++) {
      amrex::Real Rmk = Constants::RU * imw(ii);
      dpdYk[ii] = Rmk * T * InvEosT1Denom - dAmdY[ii] * InvEosT2Denom +
                  Bi[ii] * (Rm * T * InvEosT1Denom * InvEosT1Denom +
                            am * InvEosT2Denom * InvEosT3Denom);
    }

    for (int ii = 0; ii < NUM_SPECIES; ii++) {

      diP[ii] = (Y[ii] / (Rm * T) *
                 ((dAmdY[ii] - am * Bi[ii] * InvBm) * InvEosT2Denom +
                  am * Bi[ii] * InvBm * InvEosT3Denom * InvEosT3Denom)) -
                Y[ii] * (Bi[ii] * InvEosT1Denom * InvEosT1Denom +
                         wbar * imw(ii) * InvEosT1Denom);
      diP[ii] /= dpdtau;

      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dijY[ii][jj] = 0.0;
      }
      dijY[ii][ii] = wbar * imw(ii);
      for (int jj = 0; jj < NUM_SPECIES; jj++) {
        dijY[ii][jj] +=
          -diP[ii] * dpdYk[jj] -
          wbar * Y[ii] * InvEosT1Denom * (Bi[ii] * imw(jj) + Bi[jj] * imw(ii)) +
          Y[ii] * InvEosT1Denom * InvEosT1Denom * Bi[ii] * Bi[jj] +
          Y[ii] / (Rm * T) *
            ((K1 - InvEosT3Denom) * dAmdY[jj] * Bi[ii] * InvBm -
             K1 * d2AmdY2[ii][jj] +
             (K1 - InvEosT3Denom) * dAmdY[ii] * Bi[jj] * InvBm +
             ((-2.0 * am * K1 + am * InvEosT3Denom) * Bi[ii] * Bi[jj] * InvBm *
              InvBm) +
             ((am * InvEosT3Denom * InvBm +
               am * InvEosT3Denom * InvEosT3Denom) *
              Bi[ii] * Bi[jj] * InvBm));
      }
    }
  }

  template <class... Args>
  AMREX_GPU_HOST_DEVICE explicit SRK(Args... /*unused*/)
  {
    GET_CRITPARAMS_SRK(sqrtOneOverTc, sqrtAsti, Bi, Fomega);
  }

  // Constants for SRK
  static constexpr amrex::Real convCritP = 1e-4;
  static constexpr amrex::Real convCritT = 1e-6;
  static constexpr int maxIter = 2000;

  amrex::Real Bi[NUM_SPECIES];
  amrex::Real sqrtOneOverTc[NUM_SPECIES];
  amrex::Real sqrtAsti[NUM_SPECIES];
  amrex::Real Fomega[NUM_SPECIES];
  
  
  
  //iterative algorithm to compute T
  void RHY2T(
    const amrex::Real R,
    const amrex::Real H,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real& T)
  {
    // T on entry is the initial guess and is overwritten on return.
    amrex::Real Tn = T;
    AMREX_ASSERT(Tn > 0.0);
    amrex::Real dT = 100000.0;
    int nIter = 0;
    while (std::abs(dT) > convCritT && nIter < maxIter) {
      ++nIter;
      amrex::Real h = 0.0;
      RTY2H(R, Tn, Y, h);
      const amrex::Real dhdt = Calc_dhmdT(R, Tn, Y);
      dT = (h - H) / dhdt;
      Tn -= dT;
    }
    T = Tn;
    AMREX_ASSERT(nIter < maxIter);
  }




  //------------------------------------------------------------------
  // Analytical derivatives for low mach supports
  // Self-contained interface: state in (R, T, Y),
  //Formulas identical to RTY2Cp / RTY2Hi.
  //------------------------------------------------------------------

  // (dp/dT)_{rho,Y}
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dpdT(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real bm = 0.0;
    MixingRuleBm(Y, bm);
    amrex::Real dAmdT = 0.0;
    Calc_dAmdT(T, Y, dAmdT);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real Rm = Constants::RU / wbar;
    return Rm / (tau - bm) - dAmdT / (tau * (tau + bm));
  }

  // (dp/drho)_{T,Y} = -tau^2 * (dp/dtau)_{T,Y}
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dpdrho(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real Rm = Constants::RU / wbar;
    const amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    const amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    const amrex::Real dpdtau =
      -Rm * T * InvEosT1Denom * InvEosT1Denom +
      am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom;
    return -tau * tau * dpdtau;
  }

  // (dh/dT)_{rho,Y}
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dhmdT(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real bm = 0.0;
    MixingRuleBm(Y, bm);
    amrex::Real am = 0.0;
    amrex::Real dAmdT = 0.0;
    amrex::Real d2AmdT2 = 0.0;
    Calc_Am_and_derivs(T, Y, am, dAmdT, d2AmdT2);
    amrex::Real Cpig = 0.0;
    CKCPBS(T, Y, Cpig);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real K1 = (1.0 / bm) * log1p(bm * R);
    const amrex::Real Rm = Constants::RU / wbar;
    return Cpig + T * d2AmdT2 * K1 - dAmdT / (tau + bm) +
           Rm * bm / (tau - bm);
  }

  // (dh/drho)_{T,Y} = -tau^2 * (dh/dtau)_{T,Y}
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dhmdrho(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    amrex::Real dAmdT = 0.0;
    Calc_dAmdT(T, Y, dAmdT);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real Rm = Constants::RU / wbar;
    const amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    const amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    const amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    const amrex::Real dhmdtau =
      -(T * dAmdT - am) * InvEosT2Denom +
      am * InvEosT3Denom * InvEosT3Denom -
      Rm * T * bm * InvEosT1Denom * InvEosT1Denom;
    return -tau * tau * dhmdtau;
  }

  // (dp/dY_k)_{rho,T,Y_{j!=k}} — all species
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_dpdY(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real dPdY[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    amrex::Real dAmdY[NUM_SPECIES];
    Calc_dAmdY(T, Y, dAmdY);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real Rm = Constants::RU / wbar;
    const amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    const amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    const amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real Rmk = Constants::RU * imw(k);
      dPdY[k] = Rmk * T * InvEosT1Denom - dAmdY[k] * InvEosT2Denom +
                Bi[k] * (Rm * T * InvEosT1Denom * InvEosT1Denom +
                         am * InvEosT2Denom * InvEosT3Denom);
    }
  }

  // (dh/dY_k)_{rho,T,Y_{j!=k}} — all species
  // Note: xi_k = (dh/dY_k)_{p,T} =
  //   Calc_dhmdY(R,T,Y,...)[k]
  //   - (Calc_dhmdrho(R,T,Y) / Calc_dpdrho(R,T,Y)) * Calc_dpdY(R,T,Y,...)[k]
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_dhmdY(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real dhdY[NUM_SPECIES])
  {
    amrex::Real wbar = 0.0;
    CKMMWY(Y, wbar);
    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    amrex::Real dAmdT = 0.0;
    Calc_dAmdT(T, Y, dAmdT);
    amrex::Real dAmdY[NUM_SPECIES];
    Calc_dAmdY(T, Y, dAmdY);
    amrex::Real d2AmdTY[NUM_SPECIES];
    Calc_d2AmdTY(T, Y, d2AmdTY);
    amrex::Real hig[NUM_SPECIES];
    CKHMS(T, hig);
    const amrex::Real tau = 1.0 / R;
    const amrex::Real K1 = (1.0 / bm) * log1p(bm * R);
    const amrex::Real Rm = Constants::RU / wbar;
    const amrex::Real InvEosT1Denom = 1.0 / (tau - bm);
    const amrex::Real InvEosT2Denom = 1.0 / (tau * (tau + bm));
    const amrex::Real InvEosT3Denom = 1.0 / (tau + bm);
    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real Rmk = Constants::RU * imw(k);
      dhdY[k] = hig[k] + (T * d2AmdTY[k] - dAmdY[k]) * K1 -
                Bi[k] * (T * dAmdT - am) * (K1 / bm - InvEosT3Denom / bm) +
                am * Bi[k] * InvEosT3Denom * InvEosT3Denom -
                InvEosT3Denom * dAmdY[k] + Rmk * T * bm * InvEosT1Denom +
                Rm * T * Bi[k] *
                  (InvEosT1Denom + bm * InvEosT1Denom * InvEosT1Denom);
    }
  }

  // xi_k = (dh/dY_k)_{p,T,Y_{j!=k}} — all species, analytic.
  // Assembled from the single-derivative functions above:
  //   xi_k = dhmdY[k] - (dhmdrho / dpdrho) * dpdY[k]
  // (the -tau^2 chain factors cancel in the ratio; note dhmdrho/dpdrho
  // equals dhmdtau/dpdtau). Intermediates are intentionally recomputed
  // per call for modularity — this is a verification path, not a hot
  // path. Cross-checked by Calc_xi_pT_num below.
  /*AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_xi_pT(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real xi[NUM_SPECIES])
  {
    amrex::Real dpdY[NUM_SPECIES];
    Calc_dhmdY(R, T, Y, xi);
    Calc_dpdY(R, T, Y, dpdY);
    const amrex::Real ratio = Calc_dhmdrho(R, T, Y) / Calc_dpdrho(R, T, Y);
    for (int k = 0; k < NUM_SPECIES; ++k) {
      xi[k] -= ratio * dpdY[k];
    }
  }*/

  // Sandler/Reid ln(phi_k) at (rho, T, Y), k_ij = 0. Units CGS.
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void RTY2lnphi(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real lnphi[NUM_SPECIES])
  {
    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    amrex::Real wbar = 0.0;
    amrex::Real P = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    CKMMWY(Y, wbar);
    RTY2P(R, T, Y, P);

    const amrex::Real RT = Constants::RU * T;
    const amrex::Real A = am * wbar * wbar * P / (RT * RT);
    const amrex::Real B = bm * wbar * P / RT;
    const amrex::Real Z = P * wbar / (R * RT);

    amrex::Real mw[NUM_SPECIES];
    molecular_weight(mw);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real YdotA = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
      amloc[i] =
        (1.0 + Fomega[i] * (1.0 - sqrtT * sqrtOneOverTc[i])) * sqrtAsti[i];
      YdotA += Y[i] * amloc[i];
    }

    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real beta = (Bi[k] * mw[k]) / (bm * wbar);
      const amrex::Real two =
        2.0 * amloc[k] * mw[k] * YdotA / (am * wbar);
      lnphi[k] = (Z - 1.0) * beta - std::log(Z - B) -
                 (A / B) * (two - beta) * std::log1p(B / Z);
    }
  }

  // Contracted differential of Sandler/Reid ln(phi) at frozen T.
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static amrex::Real dlnphi_from_diffs(
    const amrex::Real Z,
    const amrex::Real Bterm,
    const amrex::Real A,
    const amrex::Real Q,
    const amrex::Real L,
    const amrex::Real beta,
    const amrex::Real psi,
    const amrex::Real dZ,
    const amrex::Real dB,
    const amrex::Real dA,
    const amrex::Real dbeta,
    const amrex::Real dpsi)
  {
    const amrex::Real dQ = (Bterm * dA - A * dB) / (Bterm * Bterm);
    const amrex::Real dL = (Z * dB - Bterm * dZ) / (Z * (Z + Bterm));
    return beta * dZ + (Z - 1.0) * dbeta - (dZ - dB) / (Z - Bterm) -
           psi * L * dQ - Q * L * dpsi - Q * psi * dL;
  }

  // D_ij = (d ln phi_i / d Y_j)_{T,p}, Y_j unconstrained (other Y fixed,
  // density re-solved to hold p). Chain rule from the (rho,T,Y) slices
  // of RTY2lnphi. Not RTY2transport's dijY.
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void RTY2dlnphidY(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real D[NUM_SPECIES][NUM_SPECIES])
  {
    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = 0; j < NUM_SPECIES; ++j) {
        D[i][j] = 0.0;
      }
    }

    amrex::Real am = 0.0;
    amrex::Real bm = 0.0;
    amrex::Real wbar = 0.0;
    amrex::Real P = 0.0;
    MixingRuleAmBm(T, Y, am, bm);
    CKMMWY(Y, wbar);
    RTY2P(R, T, Y, P);

    const amrex::Real RT = Constants::RU * T;
    const amrex::Real A = am * wbar * wbar * P / (RT * RT);
    const amrex::Real Bterm = bm * wbar * P / RT;
    const amrex::Real Z = P * wbar / (R * RT);
    if (Bterm <= 0.0 || Z <= Bterm || am <= 0.0) {
      return;
    }

    const amrex::Real Q = A / Bterm;
    const amrex::Real L = std::log1p(Bterm / Z);

    amrex::Real mw[NUM_SPECIES];
    molecular_weight(mw);
    const amrex::Real sqrtT = std::sqrt(T);
    amrex::Real amloc[NUM_SPECIES];
    amrex::Real YdotA = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
      amloc[i] =
        (1.0 + Fomega[i] * (1.0 - sqrtT * sqrtOneOverTc[i])) * sqrtAsti[i];
      YdotA += Y[i] * amloc[i];
    }
    if (YdotA <= 0.0) {
      return;
    }

    amrex::Real dAmdY[NUM_SPECIES];
    amrex::Real dPdY[NUM_SPECIES];
    Calc_dAmdY(T, Y, dAmdY);
    Calc_dpdY(R, T, Y, dPdY);
    const amrex::Real dPdrho = Calc_dpdrho(R, T, Y);

    amrex::Real beta[NUM_SPECIES];
    amrex::Real two[NUM_SPECIES];
    amrex::Real psi[NUM_SPECIES];
    for (int k = 0; k < NUM_SPECIES; ++k) {
      beta[k] = (Bi[k] * mw[k]) / (bm * wbar);
      two[k] = 2.0 * amloc[k] * mw[k] * YdotA / (am * wbar);
      psi[k] = two[k] - beta[k];
    }

    // (d ln phi_k / d rho)_{T,Y}: beta, psi frozen.
    const amrex::Real Z_rho = Z * (dPdrho / P - 1.0 / R);
    const amrex::Real A_rho = A * dPdrho / P;
    const amrex::Real B_rho = Bterm * dPdrho / P;
    amrex::Real dlnphi_drho[NUM_SPECIES];
    for (int k = 0; k < NUM_SPECIES; ++k) {
      dlnphi_drho[k] = dlnphi_from_diffs(
        Z, Bterm, A, Q, L, beta[k], psi[k], Z_rho, B_rho, A_rho, 0.0, 0.0);
    }

    // (d ln phi_k / d Y_j)_{rho,T} then lift to const p.
    for (int j = 0; j < NUM_SPECIES; ++j) {
      const amrex::Real wbar_Y = -wbar * wbar * imw(j);
      const amrex::Real bm_Y = Bi[j];
      const amrex::Real am_Y = dAmdY[j];
      const amrex::Real P_Y = dPdY[j];
      const amrex::Real rho_Y = -P_Y / dPdrho;

      const amrex::Real Z_Y = (P_Y * wbar + P * wbar_Y) / (R * RT);
      const amrex::Real A_Y =
        (am_Y * wbar * wbar * P + am * 2.0 * wbar * wbar_Y * P +
         am * wbar * wbar * P_Y) /
        (RT * RT);
      const amrex::Real B_Y =
        (bm_Y * wbar * P + bm * wbar_Y * P + bm * wbar * P_Y) / RT;

      for (int k = 0; k < NUM_SPECIES; ++k) {
        const amrex::Real beta_Y =
          -beta[k] * (bm_Y / bm + wbar_Y / wbar);
        const amrex::Real two_Y =
          two[k] * (-amloc[j] / YdotA + wbar * imw(j));
        const amrex::Real psi_Y = two_Y - beta_Y;
        const amrex::Real dlnphi_dY_rho = dlnphi_from_diffs(
          Z, Bterm, A, Q, L, beta[k], psi[k], Z_Y, B_Y, A_Y, beta_Y, psi_Y);
        D[k][j] = dlnphi_dY_rho + dlnphi_drho[k] * rho_Y;
      }
    }
  }

  //------------------------------------------------------------------
  // [Derivatives sandbox] Numerical (central-difference) counterparts
  // of the Calc_* partials above, for validation purposes.
  // Step policy: fd_kRel * max(|x|, floor), floors: T 1 K,
  // rho 1e-8 g/cm^3, Y 1e-3. Y kicks are unconstrained
  // (other Y_j fixed, no renormalization) — same convention as
  // SRKNumericDeriv.H. Units CGS throughout.
  //------------------------------------------------------------------

  static constexpr amrex::Real fd_kRel = 1.0e-6;
  static constexpr amrex::Real fd_kTFloor = 1.0;
  static constexpr amrex::Real fd_kRhoFloor = 1.0e-8;
  static constexpr amrex::Real fd_kYFloor = 1.0e-3;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static amrex::Real
  fd_step(const amrex::Real x, const amrex::Real floor_val)
  {
    return fd_kRel * amrex::max(std::abs(x), floor_val);
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void fd_copyY(
    const amrex::Real Yin[NUM_SPECIES], amrex::Real Yout[NUM_SPECIES])
  {
    for (int n = 0; n < NUM_SPECIES; ++n) {
      Yout[n] = Yin[n];
    }
  }

  // (dp/dT)_{rho,Y} — central difference on RTY2P
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dpdT_num(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    const amrex::Real dT = fd_step(T, fd_kTFloor);
    amrex::Real Pp = 0.0;
    amrex::Real Pm = 0.0;
    RTY2P(R, T + dT, Y, Pp);
    RTY2P(R, T - dT, Y, Pm);
    return (Pp - Pm) / (2.0 * dT);
  }

  // (dp/drho)_{T,Y} — central difference on RTY2P
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dpdrho_num(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    const amrex::Real dr = fd_step(R, fd_kRhoFloor);
    amrex::Real Pp = 0.0;
    amrex::Real Pm = 0.0;
    RTY2P(R + dr, T, Y, Pp);
    RTY2P(R - dr, T, Y, Pm);
    return (Pp - Pm) / (2.0 * dr);
  }

  // (dh/dT)_{rho,Y} — central difference on RTY2H
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dhmdT_num(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    const amrex::Real dT = fd_step(T, fd_kTFloor);
    amrex::Real hp = 0.0;
    amrex::Real hm = 0.0;
    RTY2H(R, T + dT, Y, hp);
    RTY2H(R, T - dT, Y, hm);
    return (hp - hm) / (2.0 * dT);
  }

  // (dh/drho)_{T,Y} — central difference on RTY2H
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Calc_dhmdrho_num(
    const amrex::Real R, const amrex::Real T, const amrex::Real Y[NUM_SPECIES])
  {
    const amrex::Real dr = fd_step(R, fd_kRhoFloor);
    amrex::Real hp = 0.0;
    amrex::Real hm = 0.0;
    RTY2H(R + dr, T, Y, hp);
    RTY2H(R - dr, T, Y, hm);
    return (hp - hm) / (2.0 * dr);
  }

  // (dp/dY_k)_{rho,T,Y_{j!=k}} — central differences on RTY2P,
  // all species
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_dpdY_num(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real dPdY[NUM_SPECIES])
  {
    amrex::Real Yp[NUM_SPECIES];
    amrex::Real Ym[NUM_SPECIES];
    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real dY = fd_step(Y[k], fd_kYFloor);
      fd_copyY(Y, Yp);
      fd_copyY(Y, Ym);
      Yp[k] += dY;
      Ym[k] -= dY;
      amrex::Real Pp = 0.0;
      amrex::Real Pm = 0.0;
      RTY2P(R, T, Yp, Pp);
      RTY2P(R, T, Ym, Pm);
      dPdY[k] = (Pp - Pm) / (2.0 * dY);
    }
  }

  // (dh/dY_k)_{rho,T,Y_{j!=k}} — central differences on RTY2H at
  // FIXED density (matches Calc_dhmdY, i.e. the {tau,T} derivative),
  // all species
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_dhmdY_num(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real dhdY[NUM_SPECIES])
  {
    amrex::Real Yp[NUM_SPECIES];
    amrex::Real Ym[NUM_SPECIES];
    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real dY = fd_step(Y[k], fd_kYFloor);
      fd_copyY(Y, Yp);
      fd_copyY(Y, Ym);
      Yp[k] += dY;
      Ym[k] -= dY;
      amrex::Real hp = 0.0;
      amrex::Real hm = 0.0;
      RTY2H(R, T, Yp, hp);
      RTY2H(R, T, Ym, hm);
      dhdY[k] = (hp - hm) / (2.0 * dY);
    }
  }

  // xi_k = (dh/dY_k)_{p,T,Y_{j!=k}} — central differences at FIXED
  // pressure: density is re-solved with PYT2R after each Y kick,
  // all species. Cross-check for the analytic combination
  //   Calc_dhmdY[k] - (Calc_dhmdrho / Calc_dpdrho) * Calc_dpdY[k]
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Calc_xi_pT_num(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real xi[NUM_SPECIES])
  {
    amrex::Real P = 0.0;
    RTY2P(R, T, Y, P);
    amrex::Real Yp[NUM_SPECIES];
    amrex::Real Ym[NUM_SPECIES];
    for (int k = 0; k < NUM_SPECIES; ++k) {
      const amrex::Real dY = fd_step(Y[k], fd_kYFloor);
      fd_copyY(Y, Yp);
      fd_copyY(Y, Ym);
      Yp[k] += dY;
      Ym[k] -= dY;
      amrex::Real rhop = 0.0;
      amrex::Real rhom = 0.0;
      amrex::Real hp = 0.0;
      amrex::Real hm = 0.0;
      PYT2R(P, Yp, T, rhop);
      PYT2R(P, Ym, T, rhom);
      RTY2H(rhop, T, Yp, hp);
      RTY2H(rhom, T, Ym, hm);
      xi[k] = (hp - hm) / (2.0 * dY);
    }
  }

  // D_ij = (d ln phi_i / d Y_j)_{T,p} — central differences at FIXED
  // pressure. Cross-check for RTY2dlnphidY.
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void RTY2dlnphidY_num(
    const amrex::Real R,
    const amrex::Real T,
    const amrex::Real Y[NUM_SPECIES],
    amrex::Real D[NUM_SPECIES][NUM_SPECIES])
  {
    amrex::Real P = 0.0;
    RTY2P(R, T, Y, P);
    amrex::Real Yp[NUM_SPECIES];
    amrex::Real Ym[NUM_SPECIES];
    amrex::Real lnphip[NUM_SPECIES];
    amrex::Real lnphim[NUM_SPECIES];
    for (int j = 0; j < NUM_SPECIES; ++j) {
      const amrex::Real dY = fd_step(Y[j], fd_kYFloor);
      fd_copyY(Y, Yp);
      fd_copyY(Y, Ym);
      Yp[j] += dY;
      Ym[j] -= dY;
      amrex::Real rhop = 0.0;
      amrex::Real rhom = 0.0;
      PYT2R(P, Yp, T, rhop);
      PYT2R(P, Ym, T, rhom);
      RTY2lnphi(rhop, T, Yp, lnphip);
      RTY2lnphi(rhom, T, Ym, lnphim);
      for (int i = 0; i < NUM_SPECIES; ++i) {
        D[i][j] = (lnphip[i] - lnphim[i]) / (2.0 * dY);
      }
    }
  }
};
} // namespace pele::physics::eos
#endif
