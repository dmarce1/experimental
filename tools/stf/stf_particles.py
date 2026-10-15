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
"""Generated Newtonian particle/local and multipole/particle operators.

Shares the M2L harmonic dual basis, contraction ladder, arithmetic DAG and
factoring passes. The M2P force includes every retained source rank.
"""
from fractions import Fraction as Q


def finish(A,g,outputs,signature,name,order,extra=None):
    g,signed=A.normalizeSigns(g,list(outputs.values()))
    live=g.live(root for root,sign in signed)
    signature,unused=signature.rsplit('\n',1)
    code=[signature]
    code+=g.emitConstants(live)
    code.append('\t\tPROFILE();')
    code.append(unused)
    code+=g.emit(live)
    for target,(root,sign) in zip(outputs,signed):
        if root!=g.zero:
            op='+=' if sign>0 else '-='
            code.append(f'\t\t{target} {op} {g.expr(root)};')
    code+=['\t}','};','']
    stats={'operator':name,'order':order,'arithmetic':g.counts(live),
           'accumulations':sum(root!=g.zero for root,sign in signed)}
    if extra: stats.update(extra)
    return '\n'.join(code),stats


def p2l(A,P):
    g=A.Dag()
    u=[g.symbol(x) for x in ('ux','uy','uz')]
    invr=g.symbol('inverseRadius'); ratio=g.symbol('targetRatio')
    weight=g.scale(g.mul(invr,g.symbol('mass')),-1)
    squared=[g.mul(x,x) for x in u]
    outputs={}
    for k in range(P+1):
        if k: weight=g.mul(weight,ratio)
        for a in A.indices(k):
            coefficient=weight
            for axis in range(A.NDIM):
                if a[axis]%2: coefficient=g.mul(coefficient,u[axis])
            polynomial={tuple(v//2 for v in power):g.const(c)
                        for power,c in A.radialPolynomial(0,a).items()}
            outputs[f'local[{A.index(a)}]']=g.mul(coefficient,A.horner(g,polynomial,squared))
    signature=f'''template <> class P2lKernel<{P}> {{
public:
\ttemplate <typename Real, typename Expansion>
\tstatic void add(Expansion& local, Real mass,
\t\tReal ux, Real uy, Real uz, Real inverseRadius, Real targetRatio) {{
\t\t(void) ux; (void) uy; (void) uz; (void) targetRatio;'''
    return finish(A,g,outputs,signature,'P2L',P)


def m2p(A,P,centered):
    g=A.Dag()
    u=[g.symbol(x) for x in ('ux','uy','uz')]
    invr=g.symbol('inverseRadius'); eta=g.symbol('sourceRatio')
    B=A.dualBasis(g,u,P)
    weight=g.one; scalar=[]; radial=[]; vector=[[] for _ in range(A.NDIM)]
    for n in range(P+1):
        if n: weight=g.scale(g.mul(weight,eta),2*n-1)
        if centered and n==1: continue
        M={a:g.symbol(f'source[{A.index(a)}]') for a in A.indices(n)}
        if n==0:
            value=M[(0,0,0)]
        else:
            get=A.accessor(g,M)
            first=[]
            for axis in range(A.NDIM):
                a=tuple(int(d==axis) for d in range(A.NDIM))
                first.append(g.mul(weight,g.add(*(g.mul(get(A.plus(a,b)),B[b])
                                                  for b in A.indices(n-1)))))
                vector[axis].append(first[-1])
            value=g.scale(g.add(*(g.mul(u[d],first[d]) for d in range(A.NDIM))),Q(1,n))
        scalar.append(value)
        radial.append(g.scale(value,2*n+1))
    potential=g.scale(g.mul(invr,g.add(*scalar)),-1)
    r2inv=g.mul(invr,invr); radial=g.add(*radial)
    outputs={'field[0]':potential}
    for axis in range(A.NDIM):
        outputs[f'field[{axis+1}]']=g.mul(r2inv,g.sub(g.add(*vector[axis]),g.mul(u[axis],radial)))
    signature=f'''template <> class M2pKernel<{P}, {str(centered).lower()}> {{
public:
\ttemplate <typename Real, typename Field, typename Source>
\tCOSMICTIGER_STF_M2P_HOST_DEVICE static void add(Field& field, Source const& source,
\t\tReal ux, Real uy, Real uz, Real inverseRadius, Real sourceRatio) {{
\t\t(void) sourceRatio;'''
    return finish(A,g,outputs,signature,'M2P',P,{'centered':centered})


def generateParticles(minOrder,maxOrder,algebra):
    code=['#if defined(__CUDACC__)',
          '#define COSMICTIGER_STF_M2P_HOST_DEVICE __host__ __device__',
          '#else',
          '#define COSMICTIGER_STF_M2P_HOST_DEVICE',
          '#endif', '']
    stats=[]
    for p in range(minOrder,maxOrder+1):
        text,counts=p2l(algebra,p); code.append(text); stats.append(counts)
        for centered in (True,False):
            text,counts=m2p(algebra,p,centered); code.append(text); stats.append(counts)
    code.append('#undef COSMICTIGER_STF_M2P_HOST_DEVICE')
    return code,stats
