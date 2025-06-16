//
// Copyright (c) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_fwd_hpp__
#define __fatrop_ocp_fwd_hpp__

namespace fatrop
{
    class OcpType;
    class ImplicitOcpType;
    template <typename T> struct ProblemDims;
    // template <> struct ProblemDims<OcpType>;
    // template <> struct ProblemDims<ImplicitOcpType>;
    template <typename T> struct ProblemInfo;
    // template <> struct ProblemInfo<OcpType>;
    // template <> struct ProblemInfo<ImplicitOcpType>;
    template <typename T> struct Jacobian;
    template <> struct Jacobian<OcpType>;
    template <> struct Jacobian<ImplicitOcpType>;
    template <typename T> struct Hessian;
    template <> struct Hessian<OcpType>;
    template <> struct Hessian<ImplicitOcpType>;
    template <typename T> struct PdSolverOrig;
    // template <> class PdSolverOrig<OcpType>;
    // template <> class PdSolverOrig<ImplicitOcpType>;
    template <typename ProblemType> class AugSystemSolver;
    template <> class AugSystemSolver<OcpType>;
    template <> class AugSystemSolver<ImplicitOcpType>;
    template <typename ProblemType> class PdSystemOrig;
    // template <> class PdSystemOrig<OcpType>;
    // template <> class PdSystemOrig<ImplicitOcpType>;
    template <typename ProblemType> class PdSystemResto;
    // template <> class PdSystemResto<OcpType>;
    // template <> class PdSystemResto<ImplicitOcpType>;
    template <typename T> struct PdSolverResto;
    // template <> class PdSolverResto<OcpType>;
    // template <> class PdSolverResto<ImplicitOcpType>;
} // namespace fatrop

#endif // __fatrop_ocp_fwd_hpp__