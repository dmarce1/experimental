#!/usr/bin/env python3
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
"""Generate unrolled symmetric trace-free Cartesian gravity operators.

Independent implementation of the harmonic-polynomial/partial-contraction
construction used by PKDGRAV3. No PKDGRAV source is copied. Source rank n and
local rank k satisfy n+k<=expansionOrder. The source cap can equal the local
cap, or be one less (the omitted highest source rank affects only potential).

Generation uses Python's standard library and clang-format. A rational expression DAG performs
constant folding, exact-expression CSE and dead-code elimination before C++
emission. No runtime polynomial algebra, factorials, loops or allocations occur
in the generated M2L kernels. See docs/cartesian_fmm.md for conventions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
import json
import sys
import re
import subprocess
from math import factorial, prod, comb
from itertools import permutations
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
dimensionDefinition = re.search(r'^#define\s+NDIM\s+(\d+)\s*$',
                                (projectRoot / 'cosmictiger/dimensions.hpp').read_text(), re.MULTILINE)
if dimensionDefinition is None:
    raise RuntimeError('dimensions.hpp must define NDIM as an integer literal')
NDIM = int(dimensionDefinition.group(1))
outputDirectory = projectRoot / 'cosmictiger/gravity/stf/generated'
reportPath = outputDirectory / 'operation_counts.json'


def indices(n):
    return [(x,n-x,0) for x in range(n+1)] + [(x,n-1-x,1) for x in range(n)]


def index(a):
    x,y,z=a; n=x+y+z
    return n*n+(x if z==0 else n+1+x)


def plus(a,b):
    return tuple(x+y for x,y in zip(a,b))


def fac(a):
    return prod(factorial(x) for x in a)


class Dag:
    def __init__(self):
        self.nodes=[]; self.cache={}
        self.zero=self.const(0); self.one=self.const(1)

    def node(self, op, *args):
        key=(op,)+args
        if key not in self.cache:
            self.cache[key]=len(self.nodes); self.nodes.append(key)
        return self.cache[key]

    def const(self,x): return self.node('constant',Q(x))
    def symbol(self,x): return self.node('symbol',x)
    def value(self,a): return self.nodes[a][1] if self.nodes[a][0]=='constant' else None

    def split(self,a):
        v=self.value(a)
        if v is not None: return v,self.one
        op,*args=self.nodes[a]
        if op=='*':
            for k in (0,1):
                v=self.value(args[k])
                if v is not None: return v,args[1-k]
        return Q(1),a

    def mul(self,a,b):
        # Fold a literal coefficient into another literal coefficient, but
        # preserve scaled intermediates otherwise: x*(15*C) can reuse 15*C.
        # Unconditionally extracting 15 here repeats that multiplication in
        # every later contraction that consumes C.
        ca,cb=self.value(a),self.value(b)
        if ca is not None and cb is not None: return self.const(ca*cb)
        if ca==0 or cb==0: return self.zero
        if ca==1: return b
        if cb==1: return a
        if ca is not None:
            v,c=self.split(b)
            if v!=1: return self.mul(self.const(ca*v),c)
        if cb is not None:
            v,c=self.split(a)
            if v!=1: return self.mul(self.const(cb*v),c)
        base=self.node('*',*sorted((a,b)))
        candidates=[base]
        # Reassociate only when the alternate inner product already exists.
        # Prefer fewer operations, then older shared subexpressions on ties.
        for x,y in ((a,b),(b,a)):
            if self.nodes[x][0]=='*':
                v,w=self.nodes[x][1:]
                for z,t in ((v,w),(w,v)):
                    pair=('*',)+tuple(sorted((y,z)))
                    if pair in self.cache:
                        candidates.append(self.node('*',*sorted((t,self.cache[pair]))))
        return min(candidates,key=lambda root:(sum(self.counts(self.live([root])).values()),sum(self.live([root])),root))

    def scale(self,a,v): return self.mul(self.const(v),a)

    def add(self,*terms):
        grouped={}
        for t in terms:
            c,a=self.split(t); grouped[a]=grouped.get(a,Q(0))+c
        terms=[(a,c) for a,c in sorted(grouped.items()) if c]
        if not terms: return self.zero
        common=Q(1)
        if all(c<0 for a,c in terms): common=-common
        terms=[(a,c/common) for a,c in terms]
        terms.sort(key=lambda ac: (ac[1]<0,ac[0]))
        def sumGroup(terms):
            if not terms: return self.zero
            out=self.scale(*terms[0])
            for a,c in terms[1:]: out=self.node('+',*sorted((out,self.scale(a,c))))
            return out
        positive=sumGroup([(a,c) for a,c in terms if c>0])
        negative=sumGroup([(a,-c) for a,c in terms if c<0])
        # Group a shared negative sum once, instead of subtracting its terms
        # separately in each output (e.g. the scalar radial combinations).
        out=positive if negative==self.zero else self.node('-',positive,negative)
        return self.scale(out,common)

    def sub(self,a,b): return self.add(a,self.scale(b,-1))

    def live(self, roots):
        done=set()
        def visit(a):
            if a in done: return
            done.add(a)
            if self.nodes[a][0] in ('+','-','*'):
                for b in self.nodes[a][1:]: visit(b)
        for a in roots: visit(a)
        return done

    def expr(self,a):
        node=self.nodes[a]
        if node[0]=='symbol': return node[1]
        if node[0]=='constant':
            if hasattr(self, 'constantNames') and a in self.constantNames:
                return f'Real({self.constantNames[a]})'
            q=node[1]
            if q.denominator==1:
                suffix='.0' if abs(q.numerator)>2**63-1 else ''
                return f'Real({q.numerator}{suffix})'
            return f'Real({q.numerator}.0 / {q.denominator}.0)'
        return f't{a}'

    def emitConstants(self,live,indent='\t\t'):
        """Name scalar coefficients before the generated runtime arithmetic.

        Retain the original literal type and rational evaluation so Real's
        conversions, including SIMD conversions, have unchanged semantics.
        """
        self.constantNames={}
        out=[]
        for a in sorted(live):
            if self.nodes[a][0]!='constant':
                continue
            expression=self.expr(a)[len('Real('):-1]
            name=f'kCoefficient{a}'
            self.constantNames[a]=name
            out.append(f'{indent}[[maybe_unused]] constexpr auto {name} = {expression};')
        return out

    def emit(self,live,already=frozenset(),indent='\t\t'):
        out=[]
        for a in sorted(live-already):
            op,*args=self.nodes[a]
            if op in ('+','-','*'):
                out.append(f'{indent}Real const t{a} = {self.expr(args[0])} {op} {self.expr(args[1])};')
        return out

    def counts(self,live):
        counts=Counter(self.nodes[a][0] for a in live if self.nodes[a][0] in ('+','-','*'))
        return {op:counts[op] for op in ('*','+','-') if counts[op]}


def accessor(g,components):
    @lru_cache(None)
    def get(a):
        x,y,z=a
        if z<2: return components[a]
        return g.scale(g.add(get((x+2,y,z-2)),get((x,y+2,z-2))),-1)
    return get


def dualBasis(g,u,degree):
    """Exact harmonic dual basis, factored by coordinate parity.

    Eliminating pairs of z indices from an STF contraction gives this finite
    polynomial directly. The Euler recurrence is equivalent but repeats work.
    All terms share the parity of (a,b,c), leaving a polynomial in x²,y²,z².
    """
    squared=[g.mul(x,x) for x in u]
    B={}; used=set()
    for n in range(degree+1):
        for a,b,c in indices(n):
            polynomial={}
            for kx in range(a//2+1):
                for ky in range(b//2+1):
                    k=kx+ky
                    powers=(a-2*kx,b-2*ky,c+2*k)
                    coefficient=Q((-1)**k*comb(k,kx),fac(powers))
                    polynomial[tuple(v//2 for v in powers)]=g.const(coefficient)
            prefactor=g.one
            for dim,power in enumerate((a,b,c)):
                if power%2: prefactor=g.mul(prefactor,u[dim])
            B[a,b,c]=g.mul(prefactor,horner(g,polynomial,squared,used))
            used.update(g.live([B[a,b,c]]))
    return B


def radialPolynomial(n,a):
    """Exact unit-radius derivatives of |u+t|^(-2n-1), as angular monomials.

    The Kronecker-delta pairings leave a-2b free unit-vector factors. Keeping
    these monomials symbolic lets different source ranks share their angular
    factors before emission, recovering expressions such as PKDGRAV's T0.
    """
    k=sum(a); result={}
    for bx in range(a[0]//2+1):
        for by in range(a[1]//2+1):
            for bz in range(a[2]//2+1):
                b=(bx,by,bz); pairs=sum(b)
                free=tuple(x-2*y for x,y in zip(a,b))
                radial=prod(range(2*n+1,2*n+2*(k-pairs),2))
                coefficient=Q((-1)**(k-pairs)*radial*fac(a),fac(free)*fac(b)*2**pairs)
                result[free]=coefficient
    return result


def horner(g,p,u,used=None):
    """Try all three-variable Horner orders, sharing a global expression DAG."""
    def build(poly,axes):
        if not axes: return poly.get((0,0,0),g.zero)
        axis=axes[0]; groups={}
        for a,value in poly.items():
            b=list(a); degree=b[axis]; b[axis]=0
            groups.setdefault(degree,{})[tuple(b)]=value
        degree=max(groups,default=0)
        result=build(groups.get(degree,{}),axes[1:])
        for j in range(degree-1,-1,-1):
            result=g.add(g.mul(result,u[axis]),build(groups.get(j,{}),axes[1:]))
        return result
    candidates=[build(p,axes) for axes in permutations(range(NDIM))]
    # Select by additional work across outputs, not each output in isolation.
    # Unselected trial expressions are removed by the final liveness pass.
    if used is None:
        if not hasattr(g,'horner_used'): g.horner_used=set()
        used=g.horner_used
    root=min(candidates,key=lambda a:(sum(g.counts(g.live([a])-used).values()),a))
    used.update(g.live([root]))
    return root


def buildKernel(E,centered):
    g=Dag(); u=[g.symbol('ux'),g.symbol('uy'),g.symbol('uz')]
    eta=g.symbol('sourceRatio'); v=g.symbol('targetRatio'); invr=g.symbol('inverseRadius')
    B=dualBasis(g,u,E)
    weights=[g.one]
    for n in range(1,E+1): weights.append(g.scale(g.mul(weights[-1],eta),2*n-1))
    localWeights=[g.scale(invr,-1)]
    for k in range(1,E+1): localWeights.append(g.mul(localWeights[-1],v))
    contractions={}
    top=g.zero
    for n in range(E+1):
        if centered and n==1: continue
        K=E-n; J=min(n,K)
        M={a:g.symbol(f'source[{index(a)}]') for a in indices(n)}
        get=accessor(g,M)
        C={J:{a:g.mul(weights[n],g.add(*(g.mul(get(plus(a,b)),B[b]) for b in indices(n-J)))) for a in indices(J)}}
        for j in range(J-1,-1,-1):
            get=accessor(g,C[j+1])
            C[j]={a:g.scale(g.add(*(g.mul(u[i],get(plus(a,tuple(int(i==d) for d in range(NDIM)))))
                                      for i in range(NDIM))),Q(1,n-j)) for a in indices(j)}
        if n==E and E>0:
            top=g.mul(localWeights[0],C[0][(0,0,0)])
        else:
            contractions[n]=C
    result={}
    for k in range(E+1):
        for a in indices(k):
            angular={}
            monopoleOnly = k == E or (centered and k == E-1)
            for n,C in contractions.items():
                if n+k>E: continue
                for j,components in C.items():
                    if j>k: continue
                    for b,value in components.items():
                        if any(y>x for x,y in zip(a,b)): continue
                        factor=prod(comb(x,y) for x,y in zip(a,b))
                        derivative=tuple(x-y for x,y in zip(a,b))
                        for power,coefficient in radialPolynomial(n,derivative).items():
                            angular.setdefault(power,[]).append(g.scale(g.one if monopoleOnly else value,coefficient*factor))
            polynomial={power:g.add(*terms) for power,terms in angular.items()}
            if monopoleOnly:
                # Pull the mass into the shared radial weight. Derivatives of
                # 1/r have fixed coordinate parity, so factor odd powers out
                # and evaluate the remainder in squared coordinates.
                coefficient=g.mul(localWeights[k],g.symbol('source[0]'))
                for dim in range(NDIM):
                    if a[dim]%2: coefficient=g.mul(coefficient,u[dim])
                polynomial={tuple(v//2 for v in power):c for power,c in polynomial.items()}
                result[a]=g.mul(coefficient,horner(g,polynomial,[g.mul(x,x) for x in u]))
            else:
                result[a]=g.mul(localWeights[k],horner(g,polynomial,u))
    return g,result,top


def normalizeSigns(old,roots):
    """Carry signs on edges; absorb them in subtraction and accumulation.

    In particular, x*(-y) must not emit a separate negation followed by an
    addition when the consuming expression can simply subtract x*y.
    """
    new=type(old)()
    def signedSum(a,sa,b,sb):
        if a==new.zero:return b,sb
        if b==new.zero:return a,sa
        va,vb=new.value(a),new.value(b)
        if va is not None and vb is not None:
            val=sa*va+sb*vb
            return new.const(abs(val)),(-1 if val<0 else 1)
        if sa==sb:
            return new.node('+',*sorted((a,b))),sa
        if a==b:return new.zero,1
        # Give a difference and its negation one shared node plus an edge sign.
        if a>b:a,b,sa=b,a,-sa
        return new.node('-',a,b),sa
    @lru_cache(None)
    def visit(i):
        op,*args=old.nodes[i]
        if op=='constant':
            value=args[0]
            return new.const(abs(value)),(-1 if value<0 else 1)
        if op=='symbol':return new.symbol(args[0]),1
        a,sa=visit(args[0]);b,sb=visit(args[1])
        if op=='*':return new.mul(a,b),sa*sb
        if op=='-':sb=-sb
        assert op in ('+','-')
        return signedSum(a,sa,b,sb)
    result=[visit(i) for i in roots]
    return new,result


def emitKernel(E,centered):
    g,result,top=buildKernel(E,centered)
    g,signed=normalizeSigns(g,list(result.values())+[top])
    result=dict(zip(result,signed[:-1])); top,topSign=signed[-1]
    baseLive=g.live(root for root,sign in result.values()); topLive=g.live([top])
    out=[f'template <> class M2lKernel<{E}, {str(centered).lower()}> {{',
         'public:',
         '\ttemplate <typename Real, typename Expansion, typename Source, int sourceOrder>',
         '\tCOSMICTIGER_STF_M2L_HOST_DEVICE static void add(Expansion& local, Source const& source,',
         '\t\tReal ux, Real uy, Real uz, Real inverseRadius, Real sourceRatio, Real targetRatio) {']
    out+=g.emitConstants(baseLive|topLive)
    out.append('\t\tPROFILE();')
    out.append('\t\t(void) ux; (void) uy; (void) uz; (void) sourceRatio; (void) targetRatio;')
    out+=g.emit(baseLive)
    for a,(root,sign) in result.items():
        if root!=g.zero:
            op='+=' if sign>0 else '-='
            out.append(f'\t\tlocal[{index(a)}] {op} {g.expr(root)}; // {a}')
    if E>0 and top!=g.zero:
        out.append(f'\t\tif constexpr (sourceOrder == {E}) {{')
        out+=g.emit(topLive,baseLive,'\t\t\t')
        op='+=' if topSign>0 else '-='
        out.append(f'\t\t\tlocal[0] {op} {g.expr(top)}; // highest source rank: potential only')
        out.append('\t\t}')
    out.extend(['\t}','};',''])
    stats={'expansion_order':E,'centered':centered,'base_source_order':max(0,E-1),
           'arithmetic_base':g.counts(baseLive),'arithmetic_full':g.counts(baseLive|topLive),
           'trace_extra_arithmetic':{},
           'base_accumulations':sum(root!=g.zero for root,sign in result.values()),
           'full_extra_accumulations':int(E>0 and top!=g.zero),'trace_accumulations':0}
    return '\n'.join(out),stats


def generate(minOrder=0,maxOrder=9):
    parts=['// Generated by tools/generate_stf_m2l.py. Do not edit.',
           '// Distributed under the Boost Software License, Version 1.0.',
           '#pragma once', '#include <cosmictiger/profiler.hpp>', '#include <array>', '#include <cstddef>', '',
           'namespace cosmictiger::gravity::stf::detail {',
           'template <int expansionOrder, bool centered> class M2lKernel;', '',
           '#if defined(__CUDACC__)', '#define COSMICTIGER_STF_M2L_HOST_DEVICE __host__ __device__',
           '#else', '#define COSMICTIGER_STF_M2L_HOST_DEVICE', '#endif', '']
    stats=[]
    for E in range(minOrder,maxOrder+1):
        for centered in (True,False):
            code,counts=emitKernel(E,centered); parts.append(code); stats.append(counts)
    parts.append('#undef COSMICTIGER_STF_M2L_HOST_DEVICE')
    parts.append('} // namespace cosmictiger::gravity::stf::detail\n')
    from stf.stf_particles import generateParticles
    from stf.stf_translations import generateTranslations
    particleParts,particleStats=generateParticles(minOrder,maxOrder,sys.modules[__name__])
    translationParts,translationStats=generateTranslations(minOrder,maxOrder,sys.modules[__name__])
    operators=['// Generated by tools/generate_stf_m2l.py. Do not edit.',
        '// Distributed under the Boost Software License, Version 1.0.',
        '#pragma once', '#include <cosmictiger/profiler.hpp>', '#include <array>', '#include <cstddef>', '',
        'namespace cosmictiger::gravity::stf::detail {',
        'template <int order> class M2mKernel;',
        'template <int order, bool withTrace> class L2lKernel;',
        'template <int order> class P2lKernel;',
        'template <int order, bool centered> class M2pKernel;', '']
    operators+=translationParts+particleParts
    operators.append('} // namespace cosmictiger::gravity::stf::detail\n')
    config=f'''// Generated by tools/generate_stf_m2l.py. Do not edit.
#pragma once
#define COSMICTIGER_STF_GENERATED_MIN_ORDER {minOrder}
#define COSMICTIGER_STF_GENERATED_MAX_ORDER {maxOrder}
namespace cosmictiger::gravity::stf {{
inline constexpr int kMinOrder = {minOrder};
inline constexpr int kMaxOrder = {maxOrder};
}} // namespace cosmictiger::gravity::stf
'''
    report={'min_order':minOrder,'max_order':maxOrder,
            'description':'Static emitted arithmetic counts. +, - and * each count one; signs are absorbed into arithmetic/accumulation. '
            'constant rational evaluation is compile time. Add listed output accumulations. '
            'Geometry normalization/public validation and sqrt/division are excluded. '
            'No claim of optimal instruction counts or measured hardware FLOPs.',
            'kernels':stats,'operators':translationStats+particleStats}
    return {'generated_m2l.hpp':'\n'.join(parts),'generated_operators.hpp':'\n'.join(operators),
            'generated_config.hpp':config,'operation_counts.json':json.dumps(report,indent=2)+'\n'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='fail if checked-in generated files differ')
    ap.add_argument('--min-order',dest='minOrder',type=int,default=0,help='first generated operator order (default: 0)')
    ap.add_argument('--max-order',dest='maxOrder',type=int,default=9,help='last generated operator order (default: 9)')
    ap.add_argument('--output-dir',dest='outputDir',type=Path,help='write all generated files here instead of source defaults')
    args=ap.parse_args()
    if args.minOrder<0 or args.maxOrder<args.minOrder:
        ap.error('orders must satisfy 0 <= min-order <= max-order')
    outputs=generate(args.minOrder,args.maxOrder)
    notice = (projectRoot / 'cosmictiger/gravity/stf/m2l.hpp').read_text().split('*/', 1)[0] + '*/\n'
    for name, cpp in list(outputs.items()):
        if not name.endswith('.hpp'):
            continue
        cpp = cpp.replace('std::size_t', 'size_t').replace('<cstddef>', '<stddef.h>')
        cpp = re.sub(r'\bstruct (\w+(?:<[^{}]*>)?)\s*\{', r'class \1 {\npublic:', cpp)
        cpp = re.sub(r'\bstruct (\w+);', r'class \1;', cpp)
        cpp = subprocess.run(['clang-format', '--style=file', '--assume-filename=' + str(outputDirectory / name)],
                             input=cpp, text=True, capture_output=True, check=True).stdout
        outputs[name] = notice + '\n' + cpp

    failed=[]
    for name,text in outputs.items():
        path=(args.outputDir/name if args.outputDir is not None else
              reportPath if name=='operation_counts.json' else outputDirectory/name)
        if args.check:
            if not path.exists() or path.read_text()!=text: failed.append(str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    if failed: ap.exit(1,'Stale generated files: '+', '.join(failed)+'\n')
    print('Generated files are current.' if args.check else
          f'Generated STF operators for orders {args.minOrder}..{args.maxOrder}.')


if __name__=='__main__': main()
