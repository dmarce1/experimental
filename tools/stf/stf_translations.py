# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Imported from OctotigerII; distributed under the Boost Software License, Version 1.0.
"""Generate compact Cartesian STF M2M and L2L translations.

This module uses the caller's rational arithmetic DAG and C++ emitter. All
polynomial projection, translation, and trace reduction happen in Python;
the generated equal-order kernels contain only straight-line arithmetic.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial, prod


def tensorIndices(degree):
    """Full symmetric tensor indices, used only during generation."""
    return ((x, y, degree-x-y)
            for x in range(degree+1) for y in range(degree-x+1))


def indexFactorial(index):
    return prod(factorial(x) for x in index)


def oddFactorial(n):
    return prod(range(n, 0, -2))


@lru_cache(None)
def stfMonomial(alpha):
    """Exact coefficients of H_alpha(x) = STF(x**n)_alpha.

    Pairing beta equal-coordinate indices produces |x|**(2*|beta|).
    Expanding that radius power gives an ordinary homogeneous harmonic
    polynomial, without factorial normalization of the tensor component.
    """
    n = sum(alpha)
    polynomial = {}
    for bx in range(alpha[0]//2+1):
        for by in range(alpha[1]//2+1):
            for bz in range(alpha[2]//2+1):
                beta = (bx, by, bz)
                k = sum(beta)
                free = tuple(a-2*b for a, b in zip(alpha, beta))
                coefficient = Q((-1)**k * indexFactorial(alpha)
                                * oddFactorial(2*n-2*k-1),
                                2**k * indexFactorial(beta) * indexFactorial(free)
                                * oddFactorial(2*n-1))
                for radius in tensorIndices(k):
                    powers = tuple(a+2*b for a, b in zip(free, radius))
                    value = coefficient * Q(factorial(k), indexFactorial(radius))
                    polynomial[powers] = polynomial.get(powers, Q(0)) + value
    return tuple(sorted((power, value) for power, value in polynomial.items() if value))


@lru_cache(None)
def reduceSourceIndex(index):
    """Linear combination of stored STF components for a full index."""
    x, y, z = index
    if z < 2:
        return ((index, 1),)
    result = {}
    for shifted in ((x+2, y, z-2), (x, y+2, z-2)):
        for component, coefficient in reduceSourceIndex(shifted):
            result[component] = result.get(component, 0) - coefficient
    return tuple(sorted((a, c) for a, c in result.items() if c))


@lru_cache(None)
def m2mPolynomial(alpha):
    """Coefficients of the moment functional H_alpha(ratio*y + d).

    Each homogeneous part in y is harmonic, so its contraction with raw
    source moments equals its contraction with STF source moments. The ratio
    power is the source rank and is applied once to each source component by
    the DAG builder below. Dictionary keys are (displacement power, source
    independent index).
    """
    result = {}
    for power, coefficient in stfMonomial(alpha):
        for bx in range(power[0]+1):
            for by in range(power[1]+1):
                for bz in range(power[2]+1):
                    beta = (bx, by, bz)
                    displacement = tuple(a-b for a, b in zip(power, beta))
                    value = coefficient * prod(comb(a, b) for a, b in zip(power, beta))
                    for source, factor in reduceSourceIndex(beta):
                        key = displacement, source
                        result[key] = result.get(key, Q(0)) + value*factor
    result = {key: value for key, value in result.items() if value}
    # Projection must act as the identity on an already-STF source tensor.
    zeroShift = {source: value for (power, source), value in result.items()
                  if power == (0, 0, 0)}
    assert zeroShift == {alpha: Q(1)}
    return tuple(sorted(result.items()))


def coordinatePowers(g, value, order):
    result = [g.one]
    for _ in range(order):
        result.append(g.mul(result[-1], value))
    return result


def m2mDirect(order, algebra):
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), order)
    scaled = {a: g.mul(ratio[n], g.symbol(f'source[{algebra.index(a)}]'))
              for n in range(order+1) for a in algebra.indices(n)}
    result = {}
    for n in range(order+1):
        for alpha in algebra.indices(n):
            polynomial = {}
            for (power, source), coefficient in m2mPolynomial(alpha):
                polynomial.setdefault(power, []).append(g.scale(scaled[source], coefficient))
            coefficients = {power: g.add(*terms) for power, terms in polynomial.items()}
            result[algebra.index(alpha)] = algebra.horner(g, coefficients, displacement)
    return g, result


@lru_cache(None)
def equivalentTransform(order):
    """STF components to coefficients modulo tz**2 + tx**2 + ty**2.

    Both the raw tensor component and its formal monomial reduce with the
    same integer trace map. Their product gives this small, exact Gram
    matrix. Keeping this conversion separate avoids expanding every shifted
    STF component into all source/displacement monomials.
    """
    result = {}
    for full in tensorIndices(order):
        reduced = reduceSourceIndex(full)
        for target, factor in reduced:
            for source, sourceFactor in reduced:
                key = target, source
                result[key] = result.get(key, Q(0)) + Q(factor*sourceFactor, indexFactorial(full))
    return tuple(sorted((key, value) for key, value in result.items() if value))


def m2mCompact(order, algebra):
    """Factor a shift through compact harmonic-equivalent polynomials.

    Formal-variable multiplication occurs in the quotient tz**2=-tx**2-ty**2.
    Translation is multiplication by the dual basis B(d), the reduced
    coefficients of exp(d.t). The inverse basis transform uses derivatives
    of H_alpha at zero. The same-rank source term is an exact identity and
    bypasses both transforms.
    """
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), order)
    basis = algebra.dualBasis(g, displacement, order)
    scaled = {a: g.mul(ratio[n], g.symbol(f'source[{algebra.index(a)}]'))
              for n in range(order+1) for a in algebra.indices(n)}
    equivalent = {}
    for n in range(order):
        transformed = {}
        for (target, source), coefficient in equivalentTransform(n):
            transformed.setdefault(target, []).append(g.scale(scaled[source], coefficient))
        equivalent.update({target: g.add(*terms) for target, terms in transformed.items()})
    result = {}
    for n in range(order+1):
        lower = {a: [] for a in algebra.indices(n)}
        for j in range(n):
            for alpha in algebra.indices(j):
                for beta in algebra.indices(n-j):
                    product = g.mul(equivalent[alpha], basis[beta])
                    full = tuple(a+b for a, b in zip(alpha, beta))
                    for target, coefficient in reduceSourceIndex(full):
                        lower[target].append(g.scale(product, coefficient))
        lower = {a: g.add(*terms) for a, terms in lower.items()}
        for alpha in algebra.indices(n):
            terms = [scaled[alpha]]
            for beta, coefficient in stfMonomial(alpha):
                if beta[2] < 2:
                    terms.append(g.scale(lower[beta], coefficient*indexFactorial(beta)))
            result[algebra.index(alpha)] = g.add(*terms)
    return g, result


def m2m(order, algebra):
    candidates = [m2mDirect(order, algebra), m2mCompact(order, algebra)]
    def cost(candidate):
        g, result = candidate
        normalized, signed = algebra.normalizeSigns(g, list(result.values()))
        return sum(normalized.counts(normalized.live(root for root, _ in signed)).values())
    return min(candidates, key=cost)


def l2l(order, withTrace, algebra):
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), max(order, 2 if withTrace else 0))
    basis = algebra.dualBasis(g, displacement, order)
    source = {a: g.symbol(f'source[{algebra.index(a)}]')
              for n in range(order+1) for a in algebra.indices(n)}
    get = algebra.accessor(g, source)
    result = {}
    traceIndex = (order+1)**2
    trace = g.symbol(f'source[{traceIndex}]') if withTrace else g.zero
    for n in range(order+1):
        for alpha in algebra.indices(n):
            terms = []
            for j in range(order-n+1):
                for beta in algebra.indices(j):
                    index = tuple(a+b for a, b in zip(alpha, beta))
                    terms.append(g.mul(get(index), basis[beta]))
            # The compact coefficients represent a harmonic polynomial plus
            # lambda*z^2/2. Only these two stored derivatives see that term.
            if withTrace and alpha == (0, 0, 0):
                terms.append(g.scale(g.mul(trace, g.mul(displacement[2], displacement[2])), Q(1, 2)))
            elif withTrace and alpha == (0, 0, 1):
                terms.append(g.mul(trace, displacement[2]))
            result[algebra.index(alpha)] = g.mul(ratio[n], g.add(*terms))
    if withTrace:
        result[traceIndex] = g.mul(ratio[2], trace)
    return g, result


def emit(operator, order, withTrace, algebra):
    if operator == 'M2M':
        g, result = m2m(order, algebra)
        specialization = f'M2mKernel<{order}>'
        targetType = 'Multipole'
    else:
        g, result = l2l(order, withTrace, algebra)
        specialization = f'L2lKernel<{order}, {str(withTrace).lower()}>'
        targetType = 'Expansion'
    g, signed = algebra.normalizeSigns(g, list(result.values()))
    result = dict(zip(result, signed))
    live = g.live(root for root, _ in result.values())
    lines = [f'template <> class {specialization} {{',
             'public:',
             f'\ttemplate <typename Real, typename {targetType}, typename Source>',
             f'\tstatic void add({targetType}& target, Source const& source,',
             '\t\tReal const dx, Real const dy, Real const dz, Real const scaleRatio) {']
    lines.extend(g.emitConstants(live))
    lines.append('\t\tPROFILE();')
    lines.append('\t\t(void) dx; (void) dy; (void) dz; (void) scaleRatio;')
    lines.extend(line.replace('const Real ', 'Real const ') for line in g.emit(live))
    for index, (root, sign) in result.items():
        if root != g.zero:
            operation = '+=' if sign > 0 else '-='
            lines.append(f'\t\ttarget[{index}] {operation} {g.expr(root)};')
    lines.extend(['\t}', '};', ''])
    statistics = {'operator': operator, 'order': order,
                  'arithmetic': g.counts(live),
                  'accumulations': sum(root != g.zero for root, _ in result.values())}
    if operator == 'L2L':
        statistics['with_trace'] = withTrace
    return '\n'.join(lines), statistics


def generateTranslations(minOrder, maxOrder, algebra):
    """Return C++ definitions and operation counts for the inclusive range.

    The caller emits these definitions inside its detail namespace. Traceful
    L2L requires at least quadratic order; ordinary translations also support
    orders zero and one. No fixed maximum order is imposed by this module.
    """
    if minOrder < 0 or maxOrder < minOrder:
        raise ValueError('Translation orders must form a nonnegative inclusive range')
    definitions = ['template <int order> class M2mKernel;',
                   'template <int order, bool withTrace> class L2lKernel;', '']
    statistics = []
    for order in range(minOrder, maxOrder+1):
        for operator, withTrace in [('M2M', False), ('L2L', False)] + (
                [('L2L', True)] if order >= 2 else []):
            code, counts = emit(operator, order, withTrace, algebra)
            definitions.append(code)
            statistics.append(counts)
    return definitions, statistics
