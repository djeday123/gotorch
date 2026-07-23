package cuda

// PTX-ядра, скомпилированные в строку и загружаемые через cuModuleLoadData
// при инициализации PuregoBackend. .target sm_80 - forward-compatible JIT
// покрывает всё sm_80+, включая Blackwell sm_120 (проверено goml).
//
// Схема: одномерная сетка, block=256 threads, grid = ceil(n/256).
// Формат: one statement per line - некоторые ptxas-версии не принимают
// multiple statements per line через ';' разделитель, стилево NVIDIA
// использует one-per-line во всех sample PTX. См. отчёт stage4 §PTX-format.

const r02bKernelsPTX = `
.version 8.1
.target sm_89
.address_size 64

.visible .entry add_f64(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f64 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ladd_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    ld.global.f64 %vb, [%pb];
    add.f64 %vc, %va, %vb;
    st.global.f64 [%pd], %vc;
$Ladd_f64_done:
    ret;
}

.visible .entry add_f32(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f32 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ladd_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    ld.global.f32 %vb, [%pb];
    add.f32 %vc, %va, %vb;
    st.global.f32 [%pd], %vc;
$Ladd_f32_done:
    ret;
}

.visible .entry sub_f64(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f64 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsub_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    ld.global.f64 %vb, [%pb];
    sub.f64 %vc, %va, %vb;
    st.global.f64 [%pd], %vc;
$Lsub_f64_done:
    ret;
}

.visible .entry sub_f32(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f32 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsub_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    ld.global.f32 %vb, [%pb];
    sub.f32 %vc, %va, %vb;
    st.global.f32 [%pd], %vc;
$Lsub_f32_done:
    ret;
}

.visible .entry mul_f64(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f64 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lmul_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    ld.global.f64 %vb, [%pb];
    mul.f64 %vc, %va, %vb;
    st.global.f64 [%pd], %vc;
$Lmul_f64_done:
    ret;
}

.visible .entry mul_f32(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f32 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lmul_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    ld.global.f32 %vb, [%pb];
    mul.f32 %vc, %va, %vb;
    st.global.f32 [%pd], %vc;
$Lmul_f32_done:
    ret;
}

.visible .entry div_f64(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f64 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ldiv_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    ld.global.f64 %vb, [%pb];
    div.rn.f64 %vc, %va, %vb;
    st.global.f64 [%pd], %vc;
$Ldiv_f64_done:
    ret;
}

.visible .entry div_f32(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %b, %dst, %off, %pa, %pb, %pd;
    .reg .f32 %va, %vb, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ldiv_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pb, %b, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    ld.global.f32 %vb, [%pb];
    div.rn.f32 %vc, %va, %vb;
    st.global.f32 [%pd], %vc;
$Ldiv_f32_done:
    ret;
}

.visible .entry neg_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f64 %va, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lneg_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    neg.f64 %vc, %va;
    st.global.f64 [%pd], %vc;
$Lneg_f64_done:
    ret;
}

.visible .entry neg_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lneg_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    neg.f32 %vc, %va;
    st.global.f32 [%pd], %vc;
$Lneg_f32_done:
    ret;
}

.visible .entry addscalar_f64(
    .param .u64 p_a,
    .param .f64 p_scalar,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f64 %va, %vc, %s;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.f64 %s, [p_scalar];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Laddscalar_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    add.f64 %vc, %va, %s;
    st.global.f64 [%pd], %vc;
$Laddscalar_f64_done:
    ret;
}

.visible .entry addscalar_f32(
    .param .u64 p_a,
    .param .f32 p_scalar,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %s;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.f32 %s, [p_scalar];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Laddscalar_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    add.f32 %vc, %va, %s;
    st.global.f32 [%pd], %vc;
$Laddscalar_f32_done:
    ret;
}

.visible .entry mulscalar_f64(
    .param .u64 p_a,
    .param .f64 p_scalar,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f64 %va, %vc, %s;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.f64 %s, [p_scalar];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lmulscalar_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    mul.f64 %vc, %va, %s;
    st.global.f64 [%pd], %vc;
$Lmulscalar_f64_done:
    ret;
}

.visible .entry mulscalar_f32(
    .param .u64 p_a,
    .param .f32 p_scalar,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %s;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.f32 %s, [p_scalar];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lmulscalar_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mul.f32 %vc, %va, %s;
    st.global.f32 [%pd], %vc;
$Lmulscalar_f32_done:
    ret;
}

.visible .entry exp_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %log2e;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lexp_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %vc, %va, %log2e;
    ex2.approx.f32 %vc, %vc;
    st.global.f32 [%pd], %vc;
$Lexp_f32_done:
    ret;
}

.visible .entry log_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %ln2;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Llog_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    lg2.approx.f32 %vc, %va;
    mov.f32 %ln2, 0f3F317218;
    mul.f32 %vc, %vc, %ln2;
    st.global.f32 [%pd], %vc;
$Llog_f32_done:
    ret;
}

// exp_f64: fdlibm/musl e_exp.c port. Basic path for x in [-745, 709].
.visible .entry exp_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %zero32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %twopk_bits;
    .reg .f64 %x, %invln2, %ln2H, %ln2L;
    .reg .f64 %P1, %P2, %P3, %P4, %P5;
    .reg .f64 %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %c, %y, %twopk, %result;
    .reg .s32 %k_int, %expo;
    .reg .pred %p;

    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lexp_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];

    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mov.f64 %two,    0d4000000000000000;

    // k = round(x * invln2)
    mul.f64 %k_fp, %x, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;

    // hi = x - k*ln2H
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %x, %hi;
    // lo = k*ln2L
    mul.f64 %lo, %k_fp, %ln2L;
    // r = hi - lo
    sub.f64 %r, %hi, %lo;
    // t = r*r
    mul.f64 %t, %r, %r;

    // Horner: c = P1 + t*(P2 + t*(P3 + t*(P4 + t*P5)))
    mul.f64 %c, %P5, %t;
    add.f64 %c, %c, %P4;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P3;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P2;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P1;
    // c = r - t * (P1 + ...)
    mul.f64 %c, %c, %t;
    sub.f64 %c, %r, %c;

    // y = 1 - ((lo - r*c/(2-c)) - hi)
    sub.f64 %t, %two, %c;
    mul.f64 %r, %r, %c;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;

    // twopk = 2^k via bit manipulation of FP64 exponent field
    add.s32 %expo, %k_int, 1023;
    shl.b32 %expo, %expo, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expo};
    mov.f64 %twopk, %twopk_bits;

    // result = y * twopk
    mul.f64 %result, %y, %twopk;
    st.global.f64 [%pd], %result;
$Lexp_f64_done:
    ret;
}

// log_f64: fdlibm/musl e_log.c port (basic path).
// Same schema as Go math.Log: extract k and mantissa in [sqrt(2)/2, sqrt(2)),
// polynomial in s = f/(2+f) using constants Lg1..Lg7 from fdlibm.
//
// Schema:
//   Extract bits: k = exponent(x) - 1023 with sqrt(2)/2 boundary shift
//   Reduce x to m in [sqrt(2)/2, sqrt(2))
//   f = m - 1
//   hfsq = 0.5*f*f
//   s = f / (2 + f)
//   z = s*s ;  w = z*z
//   t1 = w*(Lg2 + w*(Lg4 + w*Lg6))
//   t2 = z*(Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7)))
//   R  = t2 + t1
//   ret = s*(hfsq+R) + k*ln2_lo - hfsq + f + k*ln2_hi
// ============================================================
.visible .entry log_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %hi32, %lo32, %tmp32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %xbits, %mbits;
    .reg .f64 %x, %m, %f, %hfsq, %s, %z, %w;
    .reg .f64 %Lg1, %Lg2, %Lg3, %Lg4, %Lg5, %Lg6, %Lg7;
    .reg .f64 %ln2H, %ln2L, %half, %two, %dk;
    .reg .f64 %t1, %t2, %R, %result, %tmpfp;
    .reg .s32 %k_int;
    .reg .pred %p;

    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Llog_f64_done;

    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];

    // Constants (fdlibm)
    mov.f64 %Lg1, 0d3FE5555555555593;
    mov.f64 %Lg2, 0d3FD999999997FA04;
    mov.f64 %Lg3, 0d3FD2492494229359;
    mov.f64 %Lg4, 0d3FCC71C51D8E78AF;
    mov.f64 %Lg5, 0d3FC7466496CB03DE;
    mov.f64 %Lg6, 0d3FC39A09D078C69F;
    mov.f64 %Lg7, 0d3FC2F112DF3E5244;
    mov.f64 %ln2H, 0d3FE62E42FEE00000;
    mov.f64 %ln2L, 0d3DEA39EF35793C76;
    mov.f64 %half, 0d3FE0000000000000;
    mov.f64 %two,  0d4000000000000000;

    // Extract hi32 = high 32 bits of x (as bits).
    mov.b64 %xbits, %x;
    mov.b64 {%lo32, %hi32}, %xbits;

    // k adjustment via boundary shift:
    //   hi += 0x3ff00000 - 0x3fe6a09e = 0x00095f62
    //   k  = ((int32_t)hi >> 20) - 0x3ff
    //   hi = (hi & 0x000fffff) + 0x3fe6a09e
    add.s32 %hi32, %hi32, 0x00095F62;
    shr.s32 %k_int, %hi32, 20;
    sub.s32 %k_int, %k_int, 0x3FF;
    and.b32 %tmp32, %hi32, 0x000FFFFF;
    add.s32 %tmp32, %tmp32, 0x3FE6A09E;
    // Reassemble m: hi=%tmp32, lo unchanged
    mov.b64 %mbits, {%lo32, %tmp32};
    mov.f64 %m, %mbits;

    // f = m - 1
    mov.f64 %tmpfp, 0d3FF0000000000000;
    sub.f64 %f, %m, %tmpfp;

    // hfsq = 0.5 * f * f
    mul.f64 %hfsq, %f, %f;
    mul.f64 %hfsq, %hfsq, %half;

    // s = f / (2 + f)
    add.f64 %s, %f, %two;
    div.rn.f64 %s, %f, %s;

    // z = s*s ; w = z*z
    mul.f64 %z, %s, %s;
    mul.f64 %w, %z, %z;

    // t1 = w*(Lg2 + w*(Lg4 + w*Lg6))
    mul.f64 %t1, %w, %Lg6;
    add.f64 %t1, %t1, %Lg4;
    mul.f64 %t1, %t1, %w;
    add.f64 %t1, %t1, %Lg2;
    mul.f64 %t1, %t1, %w;

    // t2 = z*(Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7)))
    mul.f64 %t2, %w, %Lg7;
    add.f64 %t2, %t2, %Lg5;
    mul.f64 %t2, %t2, %w;
    add.f64 %t2, %t2, %Lg3;
    mul.f64 %t2, %t2, %w;
    add.f64 %t2, %t2, %Lg1;
    mul.f64 %t2, %t2, %z;

    // R = t2 + t1
    add.f64 %R, %t2, %t1;

    // dk = (double)k
    cvt.rn.f64.s32 %dk, %k_int;

    // result = s*(hfsq+R) + dk*ln2_lo - hfsq + f + dk*ln2_hi
    add.f64 %tmpfp, %hfsq, %R;
    mul.f64 %result, %s, %tmpfp;
    // + dk * ln2L
    mul.f64 %tmpfp, %dk, %ln2L;
    add.f64 %result, %result, %tmpfp;
    // - hfsq
    sub.f64 %result, %result, %hfsq;
    // + f
    add.f64 %result, %result, %f;
    // + dk * ln2H
    mul.f64 %tmpfp, %dk, %ln2H;
    add.f64 %result, %result, %tmpfp;

    st.global.f64 [%pd], %result;

$Llog_f64_done:
    ret;
}

// relu_f32: c = max(a, 0)
.visible .entry relu_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %zero;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mov.f32 %zero, 0f00000000;
    max.f32 %vc, %va, %zero;
    st.global.f32 [%pd], %vc;
$Lrelu_f32_done:
    ret;
}

// relu_f64
.visible .entry relu_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f64 %va, %vc, %zero;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    mov.f64 %zero, 0d0000000000000000;
    max.f64 %vc, %va, %zero;
    st.global.f64 [%pd], %vc;
$Lrelu_f64_done:
    ret;
}

// sigmoid_f32: c = 1 / (1 + exp(-x)) via ex2.approx.f32
.visible .entry sigmoid_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %log2e, %one, %t;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %one,   0f3F800000;
    neg.f32 %t, %va;
    mul.f32 %t, %t, %log2e;
    ex2.approx.f32 %t, %t;
    add.f32 %t, %t, %one;
    rcp.approx.f32 %vc, %t;
    st.global.f32 [%pd], %vc;
$Lsigmoid_f32_done:
    ret;
}

// tanh_f32: c = tanh(x) via aparatnyy tanh.approx.f32
.visible .entry tanh_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    tanh.approx.f32 %vc, %va;
    st.global.f32 [%pd], %vc;
$Ltanh_f32_done:
    ret;
}

// sigmoid_f64: c = 1 / (1 + exp(-x)) with inline fdlibm exp
.visible .entry sigmoid_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %zero32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %twopk_bits;
    .reg .f64 %x, %negx, %invln2, %ln2H, %ln2L;
    .reg .f64 %P1, %P2, %P3, %P4, %P5, %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %c, %y, %twopk, %expmx, %result;
    .reg .s32 %k_int, %expo;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];
    neg.f64 %negx, %x;
    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mov.f64 %two,    0d4000000000000000;
    mul.f64 %k_fp, %negx, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %negx, %hi;
    mul.f64 %lo, %k_fp, %ln2L;
    sub.f64 %r, %hi, %lo;
    mul.f64 %t, %r, %r;
    mul.f64 %c, %P5, %t;
    add.f64 %c, %c, %P4;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P3;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P2;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P1;
    mul.f64 %c, %c, %t;
    sub.f64 %c, %r, %c;
    sub.f64 %t, %two, %c;
    mul.f64 %r, %r, %c;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;
    add.s32 %expo, %k_int, 1023;
    shl.b32 %expo, %expo, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expo};
    mov.f64 %twopk, %twopk_bits;
    mul.f64 %expmx, %y, %twopk;
    add.f64 %t, %expmx, %one;
    div.rn.f64 %result, %one, %t;
    st.global.f64 [%pd], %result;
$Lsigmoid_f64_done:
    ret;
}

// tanh_f64: c = (exp(2x) - 1) / (exp(2x) + 1) with inline fdlibm exp
.visible .entry tanh_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %zero32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %twopk_bits;
    .reg .f64 %x, %twox, %invln2, %ln2H, %ln2L;
    .reg .f64 %P1, %P2, %P3, %P4, %P5, %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %c, %y, %twopk, %e2x, %num, %denom, %result;
    .reg .s32 %k_int, %expo;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];
    mov.f64 %two, 0d4000000000000000;
    mul.f64 %twox, %x, %two;
    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mul.f64 %k_fp, %twox, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %twox, %hi;
    mul.f64 %lo, %k_fp, %ln2L;
    sub.f64 %r, %hi, %lo;
    mul.f64 %t, %r, %r;
    mul.f64 %c, %P5, %t;
    add.f64 %c, %c, %P4;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P3;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P2;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P1;
    mul.f64 %c, %c, %t;
    sub.f64 %c, %r, %c;
    sub.f64 %t, %two, %c;
    mul.f64 %r, %r, %c;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;
    add.s32 %expo, %k_int, 1023;
    shl.b32 %expo, %expo, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expo};
    mov.f64 %twopk, %twopk_bits;
    mul.f64 %e2x, %y, %twopk;
    sub.f64 %num, %e2x, %one;
    add.f64 %denom, %e2x, %one;
    div.rn.f64 %result, %num, %denom;
    st.global.f64 [%pd], %result;
$Ltanh_f64_done:
    ret;
}

// relu_grad_f32: dX = (X > 0) ? dY : 0
.visible .entry relu_grad_f32(
    .param .u64 p_x,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %x, %dy, %dst, %off, %px, %pdy, %pd;
    .reg .f32 %vx, %vdy, %vc, %zero;
    .reg .pred %pmask, %p;
    ld.param.u64 %x, [p_x];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %px, %x, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vx, [%px];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %zero, 0f00000000;
    setp.gt.f32 %pmask, %vx, %zero;
    selp.f32 %vc, %vdy, %zero, %pmask;
    st.global.f32 [%pd], %vc;
$Lrelu_grad_f32_done:
    ret;
}

// relu_grad_f64
.visible .entry relu_grad_f64(
    .param .u64 p_x,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %x, %dy, %dst, %off, %px, %pdy, %pd;
    .reg .f64 %vx, %vdy, %vc, %zero;
    .reg .pred %pmask, %p;
    ld.param.u64 %x, [p_x];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %px, %x, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vx, [%px];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %zero, 0d0000000000000000;
    setp.gt.f64 %pmask, %vx, %zero;
    selp.f64 %vc, %vdy, %zero, %pmask;
    st.global.f64 [%pd], %vc;
$Lrelu_grad_f64_done:
    ret;
}

// sigmoid_grad_f32: dX = dY * Y * (1 - Y)
.visible .entry sigmoid_grad_f32(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f32 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vy, [%py];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %one, 0f3F800000;
    sub.f32 %t, %one, %vy;
    mul.f32 %t, %t, %vy;
    mul.f32 %vc, %vdy, %t;
    st.global.f32 [%pd], %vc;
$Lsigmoid_grad_f32_done:
    ret;
}

// sigmoid_grad_f64
.visible .entry sigmoid_grad_f64(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f64 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vy, [%py];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %one, 0d3FF0000000000000;
    sub.f64 %t, %one, %vy;
    mul.f64 %t, %t, %vy;
    mul.f64 %vc, %vdy, %t;
    st.global.f64 [%pd], %vc;
$Lsigmoid_grad_f64_done:
    ret;
}

// tanh_grad_f32: dX = dY * (1 - Y*Y)
.visible .entry tanh_grad_f32(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f32 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vy, [%py];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %one, 0f3F800000;
    mul.f32 %t, %vy, %vy;
    sub.f32 %t, %one, %t;
    mul.f32 %vc, %vdy, %t;
    st.global.f32 [%pd], %vc;
$Ltanh_grad_f32_done:
    ret;
}

// tanh_grad_f64
.visible .entry tanh_grad_f64(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f64 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vy, [%py];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %one, 0d3FF0000000000000;
    mul.f64 %t, %vy, %vy;
    sub.f64 %t, %one, %t;
    mul.f64 %vc, %vdy, %t;
    st.global.f64 [%pd], %vc;
$Ltanh_grad_f64_done:
    ret;
}

// sum_f64: single-block 256-thread reduction with SMEM tree reduce.
// Note: user regs must NOT be named %tid - conflict with special reg %tid.
.visible .entry sum_f64(
    .param .u64 p_a,
    .param .u64 p_out,
    .param .u32 p_n
) {
    .shared .align 8 .u64 sum_f64_sm[256];
    .reg .u32 %tidx, %n, %i, %stride, %sh_base, %sh_addr, %partner_addr;
    .reg .u64 %a, %out, %pa, %off_g;
    .reg .f64 %v, %acc, %other;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %out, [p_out];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.f64 %acc, 0d0000000000000000;
    mov.u32 %i, %tidx;
$Lsum_f64_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra $Lsum_f64_reduce;
    mul.wide.u32 %off_g, %i, 8;
    add.u64 %pa, %a, %off_g;
    ld.global.f64 %v, [%pa];
    add.f64 %acc, %acc, %v;
    add.u32 %i, %i, 256;
    bra $Lsum_f64_loop;
$Lsum_f64_reduce:
    mov.u32 %sh_base, sum_f64_sm;
    shl.b32 %sh_addr, %tidx, 3;
    add.u32 %sh_addr, %sh_addr, %sh_base;
    st.shared.f64 [%sh_addr], %acc;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsum_f64_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsum_f64_end;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsum_f64_red_skip;
    add.u32 %partner_addr, %tidx, %stride;
    shl.b32 %partner_addr, %partner_addr, 3;
    add.u32 %partner_addr, %partner_addr, %sh_base;
    ld.shared.f64 %v, [%sh_addr];
    ld.shared.f64 %other, [%partner_addr];
    add.f64 %v, %v, %other;
    st.shared.f64 [%sh_addr], %v;
$Lsum_f64_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsum_f64_red_loop;
$Lsum_f64_end:
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $Lsum_f64_ret;
    ld.shared.f64 %v, [%sh_base];
    st.global.f64 [%out], %v;
$Lsum_f64_ret:
    ret;
}

// sum_f32: F32 IO but F64 accumulator inside kernel.
.visible .entry sum_f32(
    .param .u64 p_a,
    .param .u64 p_out,
    .param .u32 p_n
) {
    .shared .align 8 .u64 sum_f32_sm[256];
    .reg .u32 %tidx, %n, %i, %stride, %sh_base, %sh_addr, %partner_addr;
    .reg .u64 %a, %out, %pa, %off_g;
    .reg .f32 %v32, %result32;
    .reg .f64 %v, %acc, %other;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %out, [p_out];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.f64 %acc, 0d0000000000000000;
    mov.u32 %i, %tidx;
$Lsum_f32_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra $Lsum_f32_reduce;
    mul.wide.u32 %off_g, %i, 4;
    add.u64 %pa, %a, %off_g;
    ld.global.f32 %v32, [%pa];
    cvt.f64.f32 %v, %v32;
    add.f64 %acc, %acc, %v;
    add.u32 %i, %i, 256;
    bra $Lsum_f32_loop;
$Lsum_f32_reduce:
    mov.u32 %sh_base, sum_f32_sm;
    shl.b32 %sh_addr, %tidx, 3;
    add.u32 %sh_addr, %sh_addr, %sh_base;
    st.shared.f64 [%sh_addr], %acc;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsum_f32_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsum_f32_end;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsum_f32_red_skip;
    add.u32 %partner_addr, %tidx, %stride;
    shl.b32 %partner_addr, %partner_addr, 3;
    add.u32 %partner_addr, %partner_addr, %sh_base;
    ld.shared.f64 %v, [%sh_addr];
    ld.shared.f64 %other, [%partner_addr];
    add.f64 %v, %v, %other;
    st.shared.f64 [%sh_addr], %v;
$Lsum_f32_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsum_f32_red_loop;
$Lsum_f32_end:
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $Lsum_f32_ret;
    ld.shared.f64 %v, [%sh_base];
    cvt.rn.f32.f64 %result32, %v;
    st.global.f32 [%out], %result32;
$Lsum_f32_ret:
    ret;
}

// softmax_f64: 1 block per row, 256 threads. 3 phases: row_max, exp+sum, divide.
// Inline fdlibm exp for numerical stability.
.visible .entry softmax_f64(
    .param .u64 p_a,
    .param .u64 p_c,
    .param .u32 p_rows,
    .param .u32 p_cols
) {
    .shared .align 8 .u64 sm_max_f64[256];
    .shared .align 8 .u64 sm_sum_f64[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems, %zero32, %expo;
    .reg .u32 %max_base, %max_addr, %max_partner;
    .reg .u32 %sum_base, %sum_addr, %sum_partner;
    .reg .u64 %a, %c, %off, %pa, %pc, %row_off_bytes;
    .reg .b64 %twopk_bits;
    .reg .f64 %v, %my_max, %row_max, %row_sum, %my_sum, %other, %xr;
    .reg .f64 %invln2, %ln2H, %ln2L, %P1, %P2, %P3, %P4, %P5, %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %cpoly, %y, %twopk, %expv;
    .reg .s32 %k_int, %expos;
    .reg .pred %p;

    ld.param.u64 %a, [p_a];
    ld.param.u64 %c, [p_c];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lsoftmax_f64_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 8;
    add.u64 %pa, %a, %row_off_bytes;
    add.u64 %pc, %c, %row_off_bytes;

    mov.u32 %max_base, sm_max_f64;
    mov.u32 %sum_base, sm_sum_f64;
    shl.b32 %max_addr, %tidx, 3;
    add.u32 %max_addr, %max_addr, %max_base;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;

    // Phase 1: row max
    mov.f64 %my_max, 0dFFF0000000000000;
    mov.u32 %j, %tidx;
$Lsm_f64_max_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsm_f64_max_reduce;
    mul.wide.u32 %off, %j, 8;
    add.u64 %pa, %pa, %off;
    ld.global.f64 %v, [%pa];
    max.f64 %my_max, %my_max, %v;
    sub.u64 %pa, %pa, %off;
    add.u32 %j, %j, 256;
    bra $Lsm_f64_max_loop;
$Lsm_f64_max_reduce:
    st.shared.f64 [%max_addr], %my_max;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsm_f64_max_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsm_f64_max_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsm_f64_max_red_skip;
    add.u32 %max_partner, %tidx, %stride;
    shl.b32 %max_partner, %max_partner, 3;
    add.u32 %max_partner, %max_partner, %max_base;
    ld.shared.f64 %v, [%max_addr];
    ld.shared.f64 %other, [%max_partner];
    max.f64 %v, %v, %other;
    st.shared.f64 [%max_addr], %v;
$Lsm_f64_max_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsm_f64_max_red_loop;
$Lsm_f64_max_red_done:
    ld.shared.f64 %row_max, [%max_base];
    bar.sync 0;

    // Phase 2: exp(x - row_max), write to c[j], sum accumulator
    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mov.f64 %two,    0d4000000000000000;

    mov.f64 %my_sum, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lsm_f64_exp_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsm_f64_sum_reduce;
    mul.wide.u32 %off, %j, 8;
    add.u64 %pa, %pa, %off;
    ld.global.f64 %v, [%pa];
    sub.u64 %pa, %pa, %off;
    sub.f64 %xr, %v, %row_max;
    // fdlibm exp(%xr) inline -> %expv
    mul.f64 %k_fp, %xr, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %xr, %hi;
    mul.f64 %lo, %k_fp, %ln2L;
    sub.f64 %r, %hi, %lo;
    mul.f64 %t, %r, %r;
    mul.f64 %cpoly, %P5, %t;
    add.f64 %cpoly, %cpoly, %P4;
    mul.f64 %cpoly, %cpoly, %t;
    add.f64 %cpoly, %cpoly, %P3;
    mul.f64 %cpoly, %cpoly, %t;
    add.f64 %cpoly, %cpoly, %P2;
    mul.f64 %cpoly, %cpoly, %t;
    add.f64 %cpoly, %cpoly, %P1;
    mul.f64 %cpoly, %cpoly, %t;
    sub.f64 %cpoly, %r, %cpoly;
    sub.f64 %t, %two, %cpoly;
    mul.f64 %r, %r, %cpoly;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;
    add.s32 %expos, %k_int, 1023;
    shl.b32 %expos, %expos, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expos};
    mov.f64 %twopk, %twopk_bits;
    mul.f64 %expv, %y, %twopk;
    add.u64 %pc, %pc, %off;
    st.global.f64 [%pc], %expv;
    sub.u64 %pc, %pc, %off;
    add.f64 %my_sum, %my_sum, %expv;
    add.u32 %j, %j, 256;
    bra $Lsm_f64_exp_loop;
$Lsm_f64_sum_reduce:
    st.shared.f64 [%sum_addr], %my_sum;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsm_f64_sum_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsm_f64_sum_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsm_f64_sum_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %v, [%sum_addr];
    ld.shared.f64 %other, [%sum_partner];
    add.f64 %v, %v, %other;
    st.shared.f64 [%sum_addr], %v;
$Lsm_f64_sum_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsm_f64_sum_red_loop;
$Lsm_f64_sum_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    bar.sync 0;

    // Phase 3: divide c[j] /= row_sum
    mov.u32 %j, %tidx;
$Lsm_f64_div_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsoftmax_f64_end;
    mul.wide.u32 %off, %j, 8;
    add.u64 %pc, %pc, %off;
    ld.global.f64 %v, [%pc];
    div.rn.f64 %v, %v, %row_sum;
    st.global.f64 [%pc], %v;
    sub.u64 %pc, %pc, %off;
    add.u32 %j, %j, 256;
    bra $Lsm_f64_div_loop;
$Lsoftmax_f64_end:
    ret;
}

// softmax_f32: same phases as softmax_f64, F32 IO + F64 sum accumulator.
.visible .entry softmax_f32(
    .param .u64 p_a,
    .param .u64 p_c,
    .param .u32 p_rows,
    .param .u32 p_cols
) {
    .shared .align 4 .u32 sm_max_f32[256];
    .shared .align 8 .u64 sm_sum_f32[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems;
    .reg .u32 %max_base, %max_addr, %max_partner;
    .reg .u32 %sum_base, %sum_addr, %sum_partner;
    .reg .u64 %a, %c, %off, %pa, %pc, %row_off_bytes;
    .reg .f32 %v32, %my_max, %row_max, %log2e, %t32, %e32, %xr, %inv_sum32;
    .reg .f32 %other32;
    .reg .f64 %my_sum, %row_sum, %e_wide, %other64, %sum_val;
    .reg .pred %p;

    ld.param.u64 %a, [p_a];
    ld.param.u64 %c, [p_c];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lsoftmax_f32_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 4;
    add.u64 %pa, %a, %row_off_bytes;
    add.u64 %pc, %c, %row_off_bytes;

    mov.u32 %max_base, sm_max_f32;
    mov.u32 %sum_base, sm_sum_f32;
    shl.b32 %max_addr, %tidx, 2;
    add.u32 %max_addr, %max_addr, %max_base;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;

    // Phase 1: row max
    mov.f32 %my_max, 0fFF800000;
    mov.u32 %j, %tidx;
$Lsm_f32_max_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsm_f32_max_reduce;
    mul.wide.u32 %off, %j, 4;
    add.u64 %pa, %pa, %off;
    ld.global.f32 %v32, [%pa];
    max.f32 %my_max, %my_max, %v32;
    sub.u64 %pa, %pa, %off;
    add.u32 %j, %j, 256;
    bra $Lsm_f32_max_loop;
$Lsm_f32_max_reduce:
    st.shared.f32 [%max_addr], %my_max;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsm_f32_max_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsm_f32_max_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsm_f32_max_red_skip;
    add.u32 %max_partner, %tidx, %stride;
    shl.b32 %max_partner, %max_partner, 2;
    add.u32 %max_partner, %max_partner, %max_base;
    ld.shared.f32 %v32, [%max_addr];
    ld.shared.f32 %other32, [%max_partner];
    max.f32 %v32, %v32, %other32;
    st.shared.f32 [%max_addr], %v32;
$Lsm_f32_max_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsm_f32_max_red_loop;
$Lsm_f32_max_red_done:
    ld.shared.f32 %row_max, [%max_base];
    bar.sync 0;

    // Phase 2: exp via ex2.approx.f32, F64 sum accumulator
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f64 %my_sum, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lsm_f32_exp_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsm_f32_sum_reduce;
    mul.wide.u32 %off, %j, 4;
    add.u64 %pa, %pa, %off;
    ld.global.f32 %v32, [%pa];
    sub.u64 %pa, %pa, %off;
    sub.f32 %xr, %v32, %row_max;
    mul.f32 %t32, %xr, %log2e;
    ex2.approx.f32 %e32, %t32;
    add.u64 %pc, %pc, %off;
    st.global.f32 [%pc], %e32;
    sub.u64 %pc, %pc, %off;
    cvt.f64.f32 %e_wide, %e32;
    add.f64 %my_sum, %my_sum, %e_wide;
    add.u32 %j, %j, 256;
    bra $Lsm_f32_exp_loop;
$Lsm_f32_sum_reduce:
    st.shared.f64 [%sum_addr], %my_sum;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lsm_f32_sum_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lsm_f32_sum_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lsm_f32_sum_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %sum_val, [%sum_addr];
    ld.shared.f64 %other64, [%sum_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%sum_addr], %sum_val;
$Lsm_f32_sum_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lsm_f32_sum_red_loop;
$Lsm_f32_sum_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    bar.sync 0;
    cvt.rn.f32.f64 %inv_sum32, %row_sum;
    rcp.approx.f32 %inv_sum32, %inv_sum32;

    // Phase 3: multiply by 1/row_sum
    mov.u32 %j, %tidx;
$Lsm_f32_div_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lsoftmax_f32_end;
    mul.wide.u32 %off, %j, 4;
    add.u64 %pc, %pc, %off;
    ld.global.f32 %v32, [%pc];
    mul.f32 %v32, %v32, %inv_sum32;
    st.global.f32 [%pc], %v32;
    sub.u64 %pc, %pc, %off;
    add.u32 %j, %j, 256;
    bra $Lsm_f32_div_loop;
$Lsoftmax_f32_end:
    ret;
}

// ================================================================
// P2-RMS: RMSNorm forward + backward, F32 and F64.
// y = gamma * x / rms, where rms = sqrt(mean(x^2) + eps).
// Backward:
//   inv_rms = 1/rms
//   dx_j    = gamma_j * dy_j * inv_rms - x_j * S * inv_rms^3 / cols
//     where S = sum_i(gamma_i * x_i * dy_i)
//   dgamma_j = sum_rows(dy * x / rms)  -- atomicAdd across rows (dgamma must be pre-zeroed).
// Schema: 1 block per row (block=256 threads), SMEM tree reduction for sum2 and S.
// F32 uses F64 accumulator for sum(x^2) (proven pattern from sum_f32/softmax_f32).
// ================================================================

// rmsnorm_f32: y = gamma * x / sqrt(mean(x^2)+eps). Row-parallel.
.visible .entry rmsnorm_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_y,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .f32 p_eps
) {
    .shared .align 8 .u64 sm_rms_sum_f32[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems, %sum_base, %sum_addr, %sum_partner;
    .reg .u64 %x, %gamma, %y, %off, %px, %py, %pg, %row_off_bytes;
    .reg .f32 %xv, %gv, %yv, %eps, %inv_rms32, %mean_sum, %fcols;
    .reg .f64 %my_sum, %row_sum, %xv_wide, %xsq, %other64, %sum_val;
    .reg .pred %p;

    ld.param.u64 %x, [p_x];
    ld.param.u64 %gamma, [p_gamma];
    ld.param.u64 %y, [p_y];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    ld.param.f32 %eps, [p_eps];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lrms_f32_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 4;
    add.u64 %px, %x, %row_off_bytes;
    add.u64 %py, %y, %row_off_bytes;

    mov.u32 %sum_base, sm_rms_sum_f32;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;

    // Phase 1: sum(x^2) per row via F64 accumulator, then SMEM tree reduction.
    mov.f64 %my_sum, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lrms_f32_sum_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lrms_f32_sum_reduce;
    mul.wide.u32 %off, %j, 4;
    add.u64 %px, %px, %off;
    ld.global.f32 %xv, [%px];
    sub.u64 %px, %px, %off;
    cvt.f64.f32 %xv_wide, %xv;
    mul.f64 %xsq, %xv_wide, %xv_wide;
    add.f64 %my_sum, %my_sum, %xsq;
    add.u32 %j, %j, 256;
    bra $Lrms_f32_sum_loop;
$Lrms_f32_sum_reduce:
    st.shared.f64 [%sum_addr], %my_sum;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lrms_f32_sum_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lrms_f32_sum_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lrms_f32_sum_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %sum_val, [%sum_addr];
    ld.shared.f64 %other64, [%sum_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%sum_addr], %sum_val;
$Lrms_f32_sum_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lrms_f32_sum_red_loop;
$Lrms_f32_sum_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    bar.sync 0;

    // Phase 2: rms = sqrt(row_sum/cols + eps); inv_rms = 1/rms in F32 via rsqrt.approx.f32.
    cvt.rn.f32.f64 %mean_sum, %row_sum;
    cvt.rn.f32.u32 %fcols, %cols;
    div.rn.f32 %mean_sum, %mean_sum, %fcols;
    add.f32 %mean_sum, %mean_sum, %eps;
    rsqrt.approx.f32 %inv_rms32, %mean_sum;

    // Phase 3: y[col] = gamma[col] * x[col] * inv_rms.
    mov.u32 %j, %tidx;
$Lrms_f32_apply_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lrms_f32_end;
    mul.wide.u32 %off, %j, 4;
    add.u64 %px, %px, %off;
    add.u64 %py, %py, %off;
    add.u64 %pg, %gamma, %off;
    ld.global.f32 %xv, [%px];
    ld.global.f32 %gv, [%pg];
    mul.f32 %yv, %xv, %inv_rms32;
    mul.f32 %yv, %yv, %gv;
    st.global.f32 [%py], %yv;
    sub.u64 %px, %px, %off;
    sub.u64 %py, %py, %off;
    add.u32 %j, %j, 256;
    bra $Lrms_f32_apply_loop;
$Lrms_f32_end:
    ret;
}

// rmsnorm_f64: F64 IO + F64 sqrt/div (no approx).
.visible .entry rmsnorm_f64(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_y,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .f64 p_eps
) {
    .shared .align 8 .u64 sm_rms_sum_f64[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems, %sum_base, %sum_addr, %sum_partner;
    .reg .u64 %x, %gamma, %y, %off, %px, %py, %pg, %row_off_bytes;
    .reg .f64 %xv, %gv, %yv, %eps, %inv_rms, %rms, %mean_sum, %fcols;
    .reg .f64 %my_sum, %row_sum, %xsq, %other64, %sum_val, %one;
    .reg .pred %p;

    ld.param.u64 %x, [p_x];
    ld.param.u64 %gamma, [p_gamma];
    ld.param.u64 %y, [p_y];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    ld.param.f64 %eps, [p_eps];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lrms_f64_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 8;
    add.u64 %px, %x, %row_off_bytes;
    add.u64 %py, %y, %row_off_bytes;

    mov.u32 %sum_base, sm_rms_sum_f64;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;

    // Phase 1: sum(x^2) per row, F64 accumulator.
    mov.f64 %my_sum, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lrms_f64_sum_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lrms_f64_sum_reduce;
    mul.wide.u32 %off, %j, 8;
    add.u64 %px, %px, %off;
    ld.global.f64 %xv, [%px];
    sub.u64 %px, %px, %off;
    mul.f64 %xsq, %xv, %xv;
    add.f64 %my_sum, %my_sum, %xsq;
    add.u32 %j, %j, 256;
    bra $Lrms_f64_sum_loop;
$Lrms_f64_sum_reduce:
    st.shared.f64 [%sum_addr], %my_sum;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lrms_f64_sum_red_loop:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lrms_f64_sum_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lrms_f64_sum_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %sum_val, [%sum_addr];
    ld.shared.f64 %other64, [%sum_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%sum_addr], %sum_val;
$Lrms_f64_sum_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lrms_f64_sum_red_loop;
$Lrms_f64_sum_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    bar.sync 0;

    // Phase 2: rms = sqrt(row_sum/cols + eps); inv_rms = 1/rms (F64, no approx).
    cvt.rn.f64.u32 %fcols, %cols;
    div.rn.f64 %mean_sum, %row_sum, %fcols;
    add.f64 %mean_sum, %mean_sum, %eps;
    sqrt.rn.f64 %rms, %mean_sum;
    mov.f64 %one, 0d3FF0000000000000;
    div.rn.f64 %inv_rms, %one, %rms;

    // Phase 3: y[col] = gamma[col] * x[col] * inv_rms.
    mov.u32 %j, %tidx;
$Lrms_f64_apply_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lrms_f64_end;
    mul.wide.u32 %off, %j, 8;
    add.u64 %px, %px, %off;
    add.u64 %py, %py, %off;
    add.u64 %pg, %gamma, %off;
    ld.global.f64 %xv, [%px];
    ld.global.f64 %gv, [%pg];
    mul.f64 %yv, %xv, %inv_rms;
    mul.f64 %yv, %yv, %gv;
    st.global.f64 [%py], %yv;
    sub.u64 %px, %px, %off;
    sub.u64 %py, %py, %off;
    add.u32 %j, %j, 256;
    bra $Lrms_f64_apply_loop;
$Lrms_f64_end:
    ret;
}

// rmsnorm_grad_f32: computes dx and dgamma. dgamma must be pre-zeroed (atomicAdd across rows).
// 1 block per row.
.visible .entry rmsnorm_grad_f32(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_dy,
    .param .u64 p_dx,
    .param .u64 p_dgamma,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .f32 p_eps
) {
    .shared .align 8 .u64 sm_grms_sum_f32[256];
    .shared .align 8 .u64 sm_grms_S_f32[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems, %sum_base, %sum_addr, %sum_partner;
    .reg .u32 %s_base, %s_addr, %s_partner;
    .reg .u64 %x, %gamma, %dy, %dx, %dgamma, %off, %px, %pdy, %pdx, %pg, %pdg, %row_off_bytes;
    .reg .f32 %xv, %gv, %dyv, %dxv, %dgv, %eps, %inv_rms32, %inv_rms3_over_cols;
    .reg .f32 %mean_sum, %fcols, %term1, %term2, %S32;
    .reg .f64 %my_sum, %row_sum, %xv_wide, %xsq, %other64, %sum_val;
    .reg .f64 %my_S, %row_S, %S_prod, %dy_wide, %g_wide;
    .reg .pred %p;

    ld.param.u64 %x, [p_x];
    ld.param.u64 %gamma, [p_gamma];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dx, [p_dx];
    ld.param.u64 %dgamma, [p_dgamma];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    ld.param.f32 %eps, [p_eps];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lgrms_f32_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 4;
    add.u64 %px, %x, %row_off_bytes;
    add.u64 %pdy, %dy, %row_off_bytes;
    add.u64 %pdx, %dx, %row_off_bytes;

    mov.u32 %sum_base, sm_grms_sum_f32;
    mov.u32 %s_base, sm_grms_S_f32;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;
    shl.b32 %s_addr, %tidx, 3;
    add.u32 %s_addr, %s_addr, %s_base;

    // Phase 1: sum(x^2) + S = sum(gamma * x * dy). Two accumulators in one loop.
    mov.f64 %my_sum, 0d0000000000000000;
    mov.f64 %my_S, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lgrms_f32_reduce_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lgrms_f32_reduce_done;
    mul.wide.u32 %off, %j, 4;
    add.u64 %px, %px, %off;
    add.u64 %pdy, %pdy, %off;
    add.u64 %pg, %gamma, %off;
    ld.global.f32 %xv, [%px];
    ld.global.f32 %dyv, [%pdy];
    ld.global.f32 %gv, [%pg];
    cvt.f64.f32 %xv_wide, %xv;
    cvt.f64.f32 %dy_wide, %dyv;
    cvt.f64.f32 %g_wide, %gv;
    mul.f64 %xsq, %xv_wide, %xv_wide;
    add.f64 %my_sum, %my_sum, %xsq;
    mul.f64 %S_prod, %g_wide, %xv_wide;
    mul.f64 %S_prod, %S_prod, %dy_wide;
    add.f64 %my_S, %my_S, %S_prod;
    sub.u64 %px, %px, %off;
    sub.u64 %pdy, %pdy, %off;
    add.u32 %j, %j, 256;
    bra $Lgrms_f32_reduce_loop;
$Lgrms_f32_reduce_done:
    st.shared.f64 [%sum_addr], %my_sum;
    st.shared.f64 [%s_addr], %my_S;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lgrms_f32_red_step:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lgrms_f32_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lgrms_f32_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %sum_val, [%sum_addr];
    ld.shared.f64 %other64, [%sum_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%sum_addr], %sum_val;
    add.u32 %s_partner, %tidx, %stride;
    shl.b32 %s_partner, %s_partner, 3;
    add.u32 %s_partner, %s_partner, %s_base;
    ld.shared.f64 %sum_val, [%s_addr];
    ld.shared.f64 %other64, [%s_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%s_addr], %sum_val;
$Lgrms_f32_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lgrms_f32_red_step;
$Lgrms_f32_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    ld.shared.f64 %row_S, [%s_base];
    bar.sync 0;

    // Phase 2: compute inv_rms and inv_rms^3 / cols.
    cvt.rn.f32.f64 %mean_sum, %row_sum;
    cvt.rn.f32.u32 %fcols, %cols;
    div.rn.f32 %mean_sum, %mean_sum, %fcols;
    add.f32 %mean_sum, %mean_sum, %eps;
    rsqrt.approx.f32 %inv_rms32, %mean_sum;
    mul.f32 %inv_rms3_over_cols, %inv_rms32, %inv_rms32;
    mul.f32 %inv_rms3_over_cols, %inv_rms3_over_cols, %inv_rms32;
    div.rn.f32 %inv_rms3_over_cols, %inv_rms3_over_cols, %fcols;
    cvt.rn.f32.f64 %S32, %row_S;

    // Phase 3: dx[j] = gamma_j*dy_j*inv_rms - x_j*S*inv_rms^3/cols;
    //          dgamma[j] += dy_j*x_j*inv_rms  (atomicAdd across rows).
    mov.u32 %j, %tidx;
$Lgrms_f32_apply_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lgrms_f32_end;
    mul.wide.u32 %off, %j, 4;
    add.u64 %px, %px, %off;
    add.u64 %pdy, %pdy, %off;
    add.u64 %pdx, %pdx, %off;
    add.u64 %pg, %gamma, %off;
    add.u64 %pdg, %dgamma, %off;
    ld.global.f32 %xv, [%px];
    ld.global.f32 %dyv, [%pdy];
    ld.global.f32 %gv, [%pg];
    // term1 = gamma_j * dy_j * inv_rms
    mul.f32 %term1, %gv, %dyv;
    mul.f32 %term1, %term1, %inv_rms32;
    // term2 = x_j * S * inv_rms^3 / cols
    mul.f32 %term2, %xv, %S32;
    mul.f32 %term2, %term2, %inv_rms3_over_cols;
    sub.f32 %dxv, %term1, %term2;
    st.global.f32 [%pdx], %dxv;
    // dgamma += dy_j * x_j * inv_rms (atomic).
    mul.f32 %dgv, %dyv, %xv;
    mul.f32 %dgv, %dgv, %inv_rms32;
    atom.global.add.f32 %dgv, [%pdg], %dgv;
    sub.u64 %px, %px, %off;
    sub.u64 %pdy, %pdy, %off;
    sub.u64 %pdx, %pdx, %off;
    add.u32 %j, %j, 256;
    bra $Lgrms_f32_apply_loop;
$Lgrms_f32_end:
    ret;
}

// rmsnorm_grad_f64: F64 version. Use sqrt.rn.f64+div.rn.f64 (no approx); atomic.add.f64.
.visible .entry rmsnorm_grad_f64(
    .param .u64 p_x,
    .param .u64 p_gamma,
    .param .u64 p_dy,
    .param .u64 p_dx,
    .param .u64 p_dgamma,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .f64 p_eps
) {
    .shared .align 8 .u64 sm_grms_sum_f64[256];
    .shared .align 8 .u64 sm_grms_S_f64[256];
    .reg .u32 %tidx, %row, %cols, %rows, %j, %stride;
    .reg .u32 %row_off_elems, %sum_base, %sum_addr, %sum_partner;
    .reg .u32 %s_base, %s_addr, %s_partner;
    .reg .u64 %x, %gamma, %dy, %dx, %dgamma, %off, %px, %pdy, %pdx, %pg, %pdg, %row_off_bytes;
    .reg .f64 %xv, %gv, %dyv, %dxv, %dgv, %eps, %inv_rms, %rms, %inv_rms3_over_cols;
    .reg .f64 %mean_sum, %fcols, %term1, %term2, %one;
    .reg .f64 %my_sum, %row_sum, %xsq, %other64, %sum_val;
    .reg .f64 %my_S, %row_S, %S_prod;
    .reg .pred %p;

    ld.param.u64 %x, [p_x];
    ld.param.u64 %gamma, [p_gamma];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dx, [p_dx];
    ld.param.u64 %dgamma, [p_dgamma];
    ld.param.u32 %rows, [p_rows];
    ld.param.u32 %cols, [p_cols];
    ld.param.f64 %eps, [p_eps];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %rows;
    @%p bra $Lgrms_f64_end;
    mov.u32 %tidx, %tid.x;

    mul.lo.u32 %row_off_elems, %row, %cols;
    mul.wide.u32 %row_off_bytes, %row_off_elems, 8;
    add.u64 %px, %x, %row_off_bytes;
    add.u64 %pdy, %dy, %row_off_bytes;
    add.u64 %pdx, %dx, %row_off_bytes;

    mov.u32 %sum_base, sm_grms_sum_f64;
    mov.u32 %s_base, sm_grms_S_f64;
    shl.b32 %sum_addr, %tidx, 3;
    add.u32 %sum_addr, %sum_addr, %sum_base;
    shl.b32 %s_addr, %tidx, 3;
    add.u32 %s_addr, %s_addr, %s_base;

    // Phase 1: sum(x^2) + S in F64.
    mov.f64 %my_sum, 0d0000000000000000;
    mov.f64 %my_S, 0d0000000000000000;
    mov.u32 %j, %tidx;
$Lgrms_f64_reduce_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lgrms_f64_reduce_done;
    mul.wide.u32 %off, %j, 8;
    add.u64 %px, %px, %off;
    add.u64 %pdy, %pdy, %off;
    add.u64 %pg, %gamma, %off;
    ld.global.f64 %xv, [%px];
    ld.global.f64 %dyv, [%pdy];
    ld.global.f64 %gv, [%pg];
    mul.f64 %xsq, %xv, %xv;
    add.f64 %my_sum, %my_sum, %xsq;
    mul.f64 %S_prod, %gv, %xv;
    mul.f64 %S_prod, %S_prod, %dyv;
    add.f64 %my_S, %my_S, %S_prod;
    sub.u64 %px, %px, %off;
    sub.u64 %pdy, %pdy, %off;
    add.u32 %j, %j, 256;
    bra $Lgrms_f64_reduce_loop;
$Lgrms_f64_reduce_done:
    st.shared.f64 [%sum_addr], %my_sum;
    st.shared.f64 [%s_addr], %my_S;
    bar.sync 0;
    mov.u32 %stride, 128;
$Lgrms_f64_red_step:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $Lgrms_f64_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $Lgrms_f64_red_skip;
    add.u32 %sum_partner, %tidx, %stride;
    shl.b32 %sum_partner, %sum_partner, 3;
    add.u32 %sum_partner, %sum_partner, %sum_base;
    ld.shared.f64 %sum_val, [%sum_addr];
    ld.shared.f64 %other64, [%sum_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%sum_addr], %sum_val;
    add.u32 %s_partner, %tidx, %stride;
    shl.b32 %s_partner, %s_partner, 3;
    add.u32 %s_partner, %s_partner, %s_base;
    ld.shared.f64 %sum_val, [%s_addr];
    ld.shared.f64 %other64, [%s_partner];
    add.f64 %sum_val, %sum_val, %other64;
    st.shared.f64 [%s_addr], %sum_val;
$Lgrms_f64_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $Lgrms_f64_red_step;
$Lgrms_f64_red_done:
    ld.shared.f64 %row_sum, [%sum_base];
    ld.shared.f64 %row_S, [%s_base];
    bar.sync 0;

    // Phase 2: rms, inv_rms in F64 (no approx).
    cvt.rn.f64.u32 %fcols, %cols;
    div.rn.f64 %mean_sum, %row_sum, %fcols;
    add.f64 %mean_sum, %mean_sum, %eps;
    sqrt.rn.f64 %rms, %mean_sum;
    mov.f64 %one, 0d3FF0000000000000;
    div.rn.f64 %inv_rms, %one, %rms;
    mul.f64 %inv_rms3_over_cols, %inv_rms, %inv_rms;
    mul.f64 %inv_rms3_over_cols, %inv_rms3_over_cols, %inv_rms;
    div.rn.f64 %inv_rms3_over_cols, %inv_rms3_over_cols, %fcols;

    // Phase 3: dx and dgamma.
    mov.u32 %j, %tidx;
$Lgrms_f64_apply_loop:
    setp.ge.u32 %p, %j, %cols;
    @%p bra $Lgrms_f64_end;
    mul.wide.u32 %off, %j, 8;
    add.u64 %px, %px, %off;
    add.u64 %pdy, %pdy, %off;
    add.u64 %pdx, %pdx, %off;
    add.u64 %pg, %gamma, %off;
    add.u64 %pdg, %dgamma, %off;
    ld.global.f64 %xv, [%px];
    ld.global.f64 %dyv, [%pdy];
    ld.global.f64 %gv, [%pg];
    mul.f64 %term1, %gv, %dyv;
    mul.f64 %term1, %term1, %inv_rms;
    mul.f64 %term2, %xv, %row_S;
    mul.f64 %term2, %term2, %inv_rms3_over_cols;
    sub.f64 %dxv, %term1, %term2;
    st.global.f64 [%pdx], %dxv;
    mul.f64 %dgv, %dyv, %xv;
    mul.f64 %dgv, %dgv, %inv_rms;
    atom.global.add.f64 %dgv, [%pdg], %dgv;
    sub.u64 %px, %px, %off;
    sub.u64 %pdy, %pdy, %off;
    sub.u64 %pdx, %pdx, %off;
    add.u32 %j, %j, 256;
    bra $Lgrms_f64_apply_loop;
$Lgrms_f64_end:
    ret;
}

// ================================================================
// P3-EMB: Embedding forward (gather) + backward (scatter-accumulate).
// Contract:
//   table [vocab, hidden] row-major, F32 or F64.
//   indices [n] int32 little-endian.
//   out [n, hidden] row-major, same dtype as table.
// Forward: out[i][d] = table[indices[i]][d]  -- pure gather, bit-exact.
// Backward: dtable[indices[i]][d] += dout[i][d]  -- atomicAdd; dtable
// MUST be pre-zeroed (Go wrapper calls cuMemsetD8 before kernel).
// Grid: 1 CTA per output row (n blocks), block=min(hidden,256).
// Undefined behavior on out-of-range index (0 <= idx < vocab is caller's
// responsibility; debug-path checks range in Go).
// ================================================================

// embedding_f32: gather rows of F32 table by int32 indices.
.visible .entry embedding_f32(
    .param .u64 p_table,
    .param .u64 p_indices,
    .param .u64 p_out,
    .param .u32 p_hidden,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %row, %hidden, %n, %d, %idx32, %woff, %ooff;
    .reg .u64 %table, %indices, %out, %addr, %off;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %table, [p_table];
    ld.param.u64 %indices, [p_indices];
    ld.param.u64 %out, [p_out];
    ld.param.u32 %hidden, [p_hidden];
    ld.param.u32 %n, [p_n];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %n;
    @%p bra $Lemb_f32_end;
    mov.u32 %tidx, %tid.x;

    // idx = indices[row]  (int32, 4-byte stride)
    mul.wide.u32 %off, %row, 4;
    add.u64 %addr, %indices, %off;
    ld.global.s32 %idx32, [%addr];

    // For each d = tidx, tidx+256, ..., < hidden: out[row][d] = table[idx][d]
    mov.u32 %d, %tidx;
$Lemb_f32_loop:
    setp.ge.u32 %p, %d, %hidden;
    @%p bra $Lemb_f32_end;
    // Source: table + (idx*hidden + d) * 4
    mul.lo.u32 %woff, %idx32, %hidden;
    add.u32 %woff, %woff, %d;
    mul.wide.u32 %off, %woff, 4;
    add.u64 %addr, %table, %off;
    ld.global.f32 %v, [%addr];
    // Dest: out + (row*hidden + d) * 4
    mul.lo.u32 %ooff, %row, %hidden;
    add.u32 %ooff, %ooff, %d;
    mul.wide.u32 %off, %ooff, 4;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %v;
    add.u32 %d, %d, 256;
    bra $Lemb_f32_loop;
$Lemb_f32_end:
    ret;
}

// embedding_f64: F64 version.
.visible .entry embedding_f64(
    .param .u64 p_table,
    .param .u64 p_indices,
    .param .u64 p_out,
    .param .u32 p_hidden,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %row, %hidden, %n, %d, %idx32, %woff, %ooff;
    .reg .u64 %table, %indices, %out, %addr, %off;
    .reg .f64 %v;
    .reg .pred %p;

    ld.param.u64 %table, [p_table];
    ld.param.u64 %indices, [p_indices];
    ld.param.u64 %out, [p_out];
    ld.param.u32 %hidden, [p_hidden];
    ld.param.u32 %n, [p_n];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %n;
    @%p bra $Lemb_f64_end;
    mov.u32 %tidx, %tid.x;

    mul.wide.u32 %off, %row, 4;
    add.u64 %addr, %indices, %off;
    ld.global.s32 %idx32, [%addr];

    mov.u32 %d, %tidx;
$Lemb_f64_loop:
    setp.ge.u32 %p, %d, %hidden;
    @%p bra $Lemb_f64_end;
    mul.lo.u32 %woff, %idx32, %hidden;
    add.u32 %woff, %woff, %d;
    mul.wide.u32 %off, %woff, 8;
    add.u64 %addr, %table, %off;
    ld.global.f64 %v, [%addr];
    mul.lo.u32 %ooff, %row, %hidden;
    add.u32 %ooff, %ooff, %d;
    mul.wide.u32 %off, %ooff, 8;
    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %v;
    add.u32 %d, %d, 256;
    bra $Lemb_f64_loop;
$Lemb_f64_end:
    ret;
}

// embedding_grad_f32: scatter-accumulate atomicAdd into dtable.
// dtable pre-zeroed by wrapper via cuMemsetD8. Collisions on repeated
// indices -> atomicAdd handles concurrency; float atomic non-associative
// -> per-run bit-drift (documented, tested).
.visible .entry embedding_grad_f32(
    .param .u64 p_indices,
    .param .u64 p_dout,
    .param .u64 p_dtable,
    .param .u32 p_hidden,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %row, %hidden, %n, %d, %idx32, %doff, %woff;
    .reg .u64 %indices, %dout, %dtable, %addr, %off;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %indices, [p_indices];
    ld.param.u64 %dout, [p_dout];
    ld.param.u64 %dtable, [p_dtable];
    ld.param.u32 %hidden, [p_hidden];
    ld.param.u32 %n, [p_n];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %n;
    @%p bra $Lembg_f32_end;
    mov.u32 %tidx, %tid.x;

    mul.wide.u32 %off, %row, 4;
    add.u64 %addr, %indices, %off;
    ld.global.s32 %idx32, [%addr];

    mov.u32 %d, %tidx;
$Lembg_f32_loop:
    setp.ge.u32 %p, %d, %hidden;
    @%p bra $Lembg_f32_end;
    // Load dout[row][d]
    mul.lo.u32 %doff, %row, %hidden;
    add.u32 %doff, %doff, %d;
    mul.wide.u32 %off, %doff, 4;
    add.u64 %addr, %dout, %off;
    ld.global.f32 %v, [%addr];
    // Atomic add into dtable[idx][d]
    mul.lo.u32 %woff, %idx32, %hidden;
    add.u32 %woff, %woff, %d;
    mul.wide.u32 %off, %woff, 4;
    add.u64 %addr, %dtable, %off;
    atom.global.add.f32 %v, [%addr], %v;
    add.u32 %d, %d, 256;
    bra $Lembg_f32_loop;
$Lembg_f32_end:
    ret;
}

// embedding_grad_f64: F64 atomic-scatter.
.visible .entry embedding_grad_f64(
    .param .u64 p_indices,
    .param .u64 p_dout,
    .param .u64 p_dtable,
    .param .u32 p_hidden,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %row, %hidden, %n, %d, %idx32, %doff, %woff;
    .reg .u64 %indices, %dout, %dtable, %addr, %off;
    .reg .f64 %v;
    .reg .pred %p;

    ld.param.u64 %indices, [p_indices];
    ld.param.u64 %dout, [p_dout];
    ld.param.u64 %dtable, [p_dtable];
    ld.param.u32 %hidden, [p_hidden];
    ld.param.u32 %n, [p_n];
    mov.u32 %row, %ctaid.x;
    setp.ge.u32 %p, %row, %n;
    @%p bra $Lembg_f64_end;
    mov.u32 %tidx, %tid.x;

    mul.wide.u32 %off, %row, 4;
    add.u64 %addr, %indices, %off;
    ld.global.s32 %idx32, [%addr];

    mov.u32 %d, %tidx;
$Lembg_f64_loop:
    setp.ge.u32 %p, %d, %hidden;
    @%p bra $Lembg_f64_end;
    mul.lo.u32 %doff, %row, %hidden;
    add.u32 %doff, %doff, %d;
    mul.wide.u32 %off, %doff, 8;
    add.u64 %addr, %dout, %off;
    ld.global.f64 %v, [%addr];
    mul.lo.u32 %woff, %idx32, %hidden;
    add.u32 %woff, %woff, %d;
    mul.wide.u32 %off, %woff, 8;
    add.u64 %addr, %dtable, %off;
    atom.global.add.f64 %v, [%addr], %v;
    add.u32 %d, %d, 256;
    bra $Lembg_f64_loop;
$Lembg_f64_end:
    ret;
}

// ================================================================
// P5A-EMB-I64: int64 -> int32 index conversion for I64 facade.
// Trivial elementwise cvt.u32.u64. Grid=ceil(n/256), block=256.
// Caller contract: 0 <= src[i] < 2^31 (else silent truncation to low 32 bits).
// ================================================================
.visible .entry cvt_u64_to_u32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %n, %idx, %v32;
    .reg .u64 %src, %dst, %v64, %addr, %off;
    .reg .pred %p;

    ld.param.u64 %src, [p_src];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_cvt_end;

    mul.wide.u32 %off, %idx, 8;
    add.u64 %addr, %src, %off;
    ld.global.u64 %v64, [%addr];
    cvt.u32.u64 %v32, %v64;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %dst, %off;
    st.global.u32 [%addr], %v32;
$L_cvt_end:
    ret;
}

// ================================================================
// P4-ROPE: Rotary Positional Embedding forward + backward, F32 (on-the-fly
// sin/cos.approx.f32) and F64 (host-precomputed cos/sin tables).
//
// Contract:
//   x, out: [batch, heads, seqLen, headDim] row-major, F32 or F64.
//   Half-layout pairs: (x[i], x[i+half]) where half=headDim/2. NOT (2i, 2i+1).
//   Forward formula:
//     angle = pos * base^(-2i/headDim)
//     dst[i]      = src[i]*cos - src[i+half]*sin
//     dst[i+half] = src[i]*sin + src[i+half]*cos
//   Backward formula (rotation by minus-angle):
//     dx[i]      = dy[i]*cos + dy[i+half]*sin
//     dx[i+half] = -dy[i]*sin + dy[i+half]*cos
//
// F32-kernel: sin/cos via .approx (matches goml.cuda.rope_f32 -> bit-exact).
// F64-kernel: cos/sin loaded from host-precomputed tables [seqLen, half] F64.
//   Rationale for tables: fdlibm F64 sin/cos = 200+ PTX lines, error-prone;
//   table O(sl*halfDim*8) = ~4MB at sl=8192/hd=128 -- acceptable.
//   Judge floor <=1e-12 held via host math.Cos/math.Sin (1 ulp F64).
// ================================================================

// rope_f32: bit-exact copy of goml.backend.cuda rope_f32 (kernels.go:314).
.visible .entry rope_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_seq_len,
    .param .u32 p_head_dim,
    .param .u32 p_num_heads,
    .param .f32 p_base
) {
    .reg .u32 %tidx, %bidx, %half_dim, %seq_len, %head_dim, %pos;
    .reg .u32 %off0, %off1;
    .reg .u64 %dst, %src, %addr;
    .reg .f32 %base, %freq, %angle, %cos_v, %sin_v;
    .reg .f32 %x0, %x1, %r0, %r1, %t0, %t1;
    .reg .f32 %fi, %fdim, %two;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %seq_len, [p_seq_len];
    ld.param.u32 %head_dim, [p_head_dim];
    ld.param.f32 %base, [p_base];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    shr.u32 %half_dim, %head_dim, 1;
    setp.ge.u32 %p, %tidx, %half_dim;
    @%p bra $L_rope_f32_done;

    rem.u32 %pos, %bidx, %seq_len;

    cvt.rn.f32.u32 %fi, %tidx;
    cvt.rn.f32.u32 %fdim, %head_dim;
    mov.f32 %two, 0f40000000;
    mul.f32 %freq, %fi, %two;
    div.approx.f32 %freq, %freq, %fdim;
    lg2.approx.f32 %t0, %base;
    mul.f32 %freq, %freq, %t0;
    neg.f32 %freq, %freq;
    ex2.approx.f32 %freq, %freq;

    cvt.rn.f32.u32 %angle, %pos;
    mul.f32 %angle, %angle, %freq;

    sin.approx.f32 %sin_v, %angle;
    cos.approx.f32 %cos_v, %angle;

    mul.lo.u32 %off0, %bidx, %head_dim;
    add.u32 %off0, %off0, %tidx;
    add.u32 %off1, %off0, %half_dim;

    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %src, %addr;
    ld.global.f32 %x0, [%addr];
    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %src, %addr;
    ld.global.f32 %x1, [%addr];

    mul.f32 %t0, %x0, %cos_v;
    mul.f32 %t1, %x1, %sin_v;
    sub.f32 %r0, %t0, %t1;
    mul.f32 %t0, %x0, %sin_v;
    mul.f32 %t1, %x1, %cos_v;
    add.f32 %r1, %t0, %t1;

    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %dst, %addr;
    st.global.f32 [%addr], %r0;
    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %dst, %addr;
    st.global.f32 [%addr], %r1;
$L_rope_f32_done:
    ret;
}

// rope_grad_f32: rotation by minus-angle.
// dx[i]      = dy[i]*cos + dy[i+half]*sin
// dx[i+half] = -dy[i]*sin + dy[i+half]*cos
.visible .entry rope_grad_f32(
    .param .u64 p_dx,
    .param .u64 p_dy,
    .param .u32 p_seq_len,
    .param .u32 p_head_dim,
    .param .u32 p_num_heads,
    .param .f32 p_base
) {
    .reg .u32 %tidx, %bidx, %half_dim, %seq_len, %head_dim, %pos;
    .reg .u32 %off0, %off1;
    .reg .u64 %dx, %dy, %addr;
    .reg .f32 %base, %freq, %angle, %cos_v, %sin_v;
    .reg .f32 %y0, %y1, %r0, %r1, %t0, %t1;
    .reg .f32 %fi, %fdim, %two;
    .reg .pred %p;

    ld.param.u64 %dx, [p_dx];
    ld.param.u64 %dy, [p_dy];
    ld.param.u32 %seq_len, [p_seq_len];
    ld.param.u32 %head_dim, [p_head_dim];
    ld.param.f32 %base, [p_base];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    shr.u32 %half_dim, %head_dim, 1;
    setp.ge.u32 %p, %tidx, %half_dim;
    @%p bra $L_rope_grad_f32_done;

    rem.u32 %pos, %bidx, %seq_len;

    cvt.rn.f32.u32 %fi, %tidx;
    cvt.rn.f32.u32 %fdim, %head_dim;
    mov.f32 %two, 0f40000000;
    mul.f32 %freq, %fi, %two;
    div.approx.f32 %freq, %freq, %fdim;
    lg2.approx.f32 %t0, %base;
    mul.f32 %freq, %freq, %t0;
    neg.f32 %freq, %freq;
    ex2.approx.f32 %freq, %freq;

    cvt.rn.f32.u32 %angle, %pos;
    mul.f32 %angle, %angle, %freq;

    sin.approx.f32 %sin_v, %angle;
    cos.approx.f32 %cos_v, %angle;

    mul.lo.u32 %off0, %bidx, %head_dim;
    add.u32 %off0, %off0, %tidx;
    add.u32 %off1, %off0, %half_dim;

    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %dy, %addr;
    ld.global.f32 %y0, [%addr];
    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %dy, %addr;
    ld.global.f32 %y1, [%addr];

    // dx[i] = y0*cos + y1*sin
    mul.f32 %t0, %y0, %cos_v;
    mul.f32 %t1, %y1, %sin_v;
    add.f32 %r0, %t0, %t1;
    // dx[i+half] = -y0*sin + y1*cos
    mul.f32 %t0, %y0, %sin_v;
    mul.f32 %t1, %y1, %cos_v;
    sub.f32 %r1, %t1, %t0;

    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %dx, %addr;
    st.global.f32 [%addr], %r0;
    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %dx, %addr;
    st.global.f32 [%addr], %r1;
$L_rope_grad_f32_done:
    ret;
}

// rope_f64: cos/sin loaded from host-precomputed tables [seqLen, half_dim] F64.
// Delivers F64 accuracy 1 ulp (host math.Cos/Sin).
.visible .entry rope_f64(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u64 p_cos,
    .param .u64 p_sin,
    .param .u32 p_seq_len,
    .param .u32 p_head_dim,
    .param .u32 p_num_heads
) {
    .reg .u32 %tidx, %bidx, %half_dim, %seq_len, %head_dim, %pos;
    .reg .u32 %off0, %off1, %tblOff;
    .reg .u64 %dst, %src, %cos_p, %sin_p, %addr;
    .reg .f64 %x0, %x1, %r0, %r1, %cos_v, %sin_v, %t0, %t1;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u64 %cos_p, [p_cos];
    ld.param.u64 %sin_p, [p_sin];
    ld.param.u32 %seq_len, [p_seq_len];
    ld.param.u32 %head_dim, [p_head_dim];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    shr.u32 %half_dim, %head_dim, 1;
    setp.ge.u32 %p, %tidx, %half_dim;
    @%p bra $L_rope_f64_done;

    rem.u32 %pos, %bidx, %seq_len;

    // Table offset = pos * half_dim + tid
    mul.lo.u32 %tblOff, %pos, %half_dim;
    add.u32 %tblOff, %tblOff, %tidx;
    mul.wide.u32 %addr, %tblOff, 8;
    add.u64 %addr, %cos_p, %addr;
    ld.global.f64 %cos_v, [%addr];
    mul.wide.u32 %addr, %tblOff, 8;
    add.u64 %addr, %sin_p, %addr;
    ld.global.f64 %sin_v, [%addr];

    mul.lo.u32 %off0, %bidx, %head_dim;
    add.u32 %off0, %off0, %tidx;
    add.u32 %off1, %off0, %half_dim;

    mul.wide.u32 %addr, %off0, 8;
    add.u64 %addr, %src, %addr;
    ld.global.f64 %x0, [%addr];
    mul.wide.u32 %addr, %off1, 8;
    add.u64 %addr, %src, %addr;
    ld.global.f64 %x1, [%addr];

    mul.f64 %t0, %x0, %cos_v;
    mul.f64 %t1, %x1, %sin_v;
    sub.f64 %r0, %t0, %t1;
    mul.f64 %t0, %x0, %sin_v;
    mul.f64 %t1, %x1, %cos_v;
    add.f64 %r1, %t0, %t1;

    mul.wide.u32 %addr, %off0, 8;
    add.u64 %addr, %dst, %addr;
    st.global.f64 [%addr], %r0;
    mul.wide.u32 %addr, %off1, 8;
    add.u64 %addr, %dst, %addr;
    st.global.f64 [%addr], %r1;
$L_rope_f64_done:
    ret;
}

// rope_grad_f64: backward via host cos/sin tables.
.visible .entry rope_grad_f64(
    .param .u64 p_dx,
    .param .u64 p_dy,
    .param .u64 p_cos,
    .param .u64 p_sin,
    .param .u32 p_seq_len,
    .param .u32 p_head_dim,
    .param .u32 p_num_heads
) {
    .reg .u32 %tidx, %bidx, %half_dim, %seq_len, %head_dim, %pos;
    .reg .u32 %off0, %off1, %tblOff;
    .reg .u64 %dx, %dy, %cos_p, %sin_p, %addr;
    .reg .f64 %y0, %y1, %r0, %r1, %cos_v, %sin_v, %t0, %t1;
    .reg .pred %p;

    ld.param.u64 %dx, [p_dx];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %cos_p, [p_cos];
    ld.param.u64 %sin_p, [p_sin];
    ld.param.u32 %seq_len, [p_seq_len];
    ld.param.u32 %head_dim, [p_head_dim];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    shr.u32 %half_dim, %head_dim, 1;
    setp.ge.u32 %p, %tidx, %half_dim;
    @%p bra $L_rope_grad_f64_done;

    rem.u32 %pos, %bidx, %seq_len;

    mul.lo.u32 %tblOff, %pos, %half_dim;
    add.u32 %tblOff, %tblOff, %tidx;
    mul.wide.u32 %addr, %tblOff, 8;
    add.u64 %addr, %cos_p, %addr;
    ld.global.f64 %cos_v, [%addr];
    mul.wide.u32 %addr, %tblOff, 8;
    add.u64 %addr, %sin_p, %addr;
    ld.global.f64 %sin_v, [%addr];

    mul.lo.u32 %off0, %bidx, %head_dim;
    add.u32 %off0, %off0, %tidx;
    add.u32 %off1, %off0, %half_dim;

    mul.wide.u32 %addr, %off0, 8;
    add.u64 %addr, %dy, %addr;
    ld.global.f64 %y0, [%addr];
    mul.wide.u32 %addr, %off1, 8;
    add.u64 %addr, %dy, %addr;
    ld.global.f64 %y1, [%addr];

    // dx[i] = y0*cos + y1*sin
    mul.f64 %t0, %y0, %cos_v;
    mul.f64 %t1, %y1, %sin_v;
    add.f64 %r0, %t0, %t1;
    // dx[i+half] = -y0*sin + y1*cos
    mul.f64 %t0, %y0, %sin_v;
    mul.f64 %t1, %y1, %cos_v;
    sub.f64 %r1, %t1, %t0;

    mul.wide.u32 %addr, %off0, 8;
    add.u64 %addr, %dx, %addr;
    st.global.f64 [%addr], %r0;
    mul.wide.u32 %addr, %off1, 8;
    add.u64 %addr, %dx, %addr;
    st.global.f64 [%addr], %r1;
$L_rope_grad_f64_done:
    ret;
}

// ================================================================
// B-impl-2: F32 <-> F16 conversion kernels for FP16 mixed precision.
// F16 = IEEE 754 binary16, stored as uint16 little-endian in DeviceBuffer.
// PTX cvt.rn.f16.f32 (F32 -> F16 round-nearest), cvt.f32.f16 (widening).
// ================================================================
.visible .entry cvt_f32_to_f16(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %n, %idx;
    .reg .u64 %src, %dst, %addr, %off;
    .reg .f32 %v32;
    .reg .b16 %v16;
    .reg .pred %p;

    ld.param.u64 %src, [p_src];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_cvt_f32_f16_end;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %src, %off;
    ld.global.f32 %v32, [%addr];
    cvt.rn.f16.f32 %v16, %v32;

    mul.wide.u32 %off, %idx, 2;
    add.u64 %addr, %dst, %off;
    st.global.b16 [%addr], %v16;
$L_cvt_f32_f16_end:
    ret;
}

.visible .entry cvt_f16_to_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %n, %idx;
    .reg .u64 %src, %dst, %addr, %off;
    .reg .f32 %v32;
    .reg .b16 %v16;
    .reg .pred %p;

    ld.param.u64 %src, [p_src];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_cvt_f16_f32_end;

    mul.wide.u32 %off, %idx, 2;
    add.u64 %addr, %src, %off;
    ld.global.b16 %v16, [%addr];
    cvt.f32.f16 %v32, %v16;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %dst, %off;
    st.global.f32 [%addr], %v32;
$L_cvt_f16_f32_end:
    ret;
}

// ================================================================
// B-impl-3: FP8 E4M3 quantize/dequantize.
//
// quantize_f32_to_f8e4m3(src, dst, scale, amax, n):
//   phase 1: reduce absmax(src) into single float, use SMEM tree reduction
//            (pattern from sum_f32). Only ONE block for simplicity (n<=65536
// covers production shapes). For larger n, caller invokes multiple times
// and aggregates host-side.
//   phase 2: scale = amax / FP8_E4M3_MAX (=448.0), write to scale device ptr.
//   phase 3: dst[i] = cvt.rn.satfinite.e4m3x2.f32 (src[i] / scale).
//
// FP8 E4M3 max = 448 (1.0*2^(15-7)=?, actually 1.111000_2 * 2^8 = 448).
// Uses PTX cvt.rn.satfinite.e4m3x2.f32 (converts 2 F32 -> 2 E4M3 packed).
//
// Grid: 1 block for phase 1-2 (all data covered by loop); block=256.
// After amax+scale, block=256 grid=ceil(n/512) for phase 3 (packed 2-per-thread).
// In this simplified stage: single grid=1 block=256 kernel, works for
// n <= threshold. Production -- separate TZ.
// ================================================================
.visible .entry quantize_f32_to_f8e4m3(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_scale,
    .param .u64 p_amax,
    .param .u32 p_n
) {
    .shared .align 4 .f32 sm_amax[256];
    .reg .u32 %tidx, %n, %j, %stride, %addr_smem;
    .reg .u32 %addr_smem_partner;
    .reg .u64 %src, %dst, %scale_p, %amax_p, %addr, %off;
    .reg .f32 %v, %av, %my_amax, %row_amax, %scale, %fp8max, %eps;
    .reg .f32 %other, %inv_scale;
    .reg .b16 %packed_hi16;
    .reg .pred %p;

    ld.param.u64 %src, [p_src];
    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %scale_p, [p_scale];
    ld.param.u64 %amax_p, [p_amax];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;

    // Phase 1: per-thread accumulate absmax.
    mov.f32 %my_amax, 0f00000000;
    mov.u32 %j, %tidx;
$L_q_loop:
    setp.ge.u32 %p, %j, %n;
    @%p bra $L_q_reduce;
    mul.wide.u32 %off, %j, 4;
    add.u64 %addr, %src, %off;
    ld.global.f32 %v, [%addr];
    abs.f32 %av, %v;
    max.f32 %my_amax, %my_amax, %av;
    add.u32 %j, %j, 256;
    bra $L_q_loop;

$L_q_reduce:
    // SMEM tree reduction for amax.
    shl.b32 %addr_smem, %tidx, 2;
    mov.u32 %stride, sm_amax;
    add.u32 %addr_smem, %addr_smem, %stride;
    st.shared.f32 [%addr_smem], %my_amax;
    bar.sync 0;
    mov.u32 %stride, 128;
$L_q_red_step:
    setp.eq.u32 %p, %stride, 0;
    @%p bra $L_q_red_done;
    setp.ge.u32 %p, %tidx, %stride;
    @%p bra $L_q_red_skip;
    add.u32 %addr_smem_partner, %tidx, %stride;
    shl.b32 %addr_smem_partner, %addr_smem_partner, 2;
    mov.u32 %j, sm_amax;
    add.u32 %addr_smem_partner, %addr_smem_partner, %j;
    ld.shared.f32 %row_amax, [%addr_smem];
    ld.shared.f32 %other, [%addr_smem_partner];
    max.f32 %row_amax, %row_amax, %other;
    st.shared.f32 [%addr_smem], %row_amax;
$L_q_red_skip:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra $L_q_red_step;
$L_q_red_done:
    // Read final amax from sm_amax[0].
    mov.u32 %j, sm_amax;
    ld.shared.f32 %row_amax, [%j];
    bar.sync 0;

    // Phase 2: scale = amax / 448.  If amax < 1e-12 -> scale = 1.0.
    // Write scale + amax to output (thread 0 only).
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $L_q_after_scale;
    mov.f32 %fp8max, 0f43E00000;   // 448.0
    mov.f32 %eps, 0f00000000;      // 0.0 sentinel
    mov.f32 %eps, 0f2ACBCCCC;      // 1e-13
    setp.lt.f32 %p, %row_amax, %eps;
    @%p mov.f32 %scale, 0f3F800000; // 1.0
    @!%p div.rn.f32 %scale, %row_amax, %fp8max;
    st.global.f32 [%scale_p], %scale;
    st.global.f32 [%amax_p], %row_amax;
$L_q_after_scale:
    bar.sync 0;

    // Broadcast scale via SMEM (thread 0 wrote it).
    // Broadcast scale via global read.
    ld.global.f32 %scale, [%scale_p];
    // inv_scale = 1 / scale. div.approx.f32 is enough (F8 tolerance).
    mov.f32 %inv_scale, 0f3F800000;
    div.approx.f32 %inv_scale, %inv_scale, %scale;

    // Phase 3: dst[i] = cvt.rn.satfinite.e4m3.f32 (src[i] * inv_scale).
    // Each thread writes 1 elem to keep code simple.
    mov.u32 %j, %tidx;
$L_q_write_loop:
    setp.ge.u32 %p, %j, %n;
    @%p bra $L_q_write_end;
    mul.wide.u32 %off, %j, 4;
    add.u64 %addr, %src, %off;
    ld.global.f32 %v, [%addr];
    mul.f32 %v, %v, %inv_scale;
    // cvt.rn.satfinite.e4m3x2.f32 -- packs 2 f32 into 16-bit hold. We use one
    // input twice (both lanes same); write lower byte.
    cvt.rn.satfinite.e4m3x2.f32 %packed_hi16, %v, %v;
    // Extract low 8-bit and write.
    mul.wide.u32 %off, %j, 1;
    add.u64 %addr, %dst, %off;
    st.global.b8 [%addr], %packed_hi16;
    add.u32 %j, %j, 256;
    bra $L_q_write_loop;
$L_q_write_end:
    ret;
}

// cast_f8e4m3_to_f32(src, dst, scale, n):
// dst[i] = f8_to_f32(src[i]) * scale[0]. dequantize with per-tensor scale.
// PTX cvt.rn.f32.e4m3x2 unpacks single f8 pair -> f32 pair; we handle scalar case.
.visible .entry cast_f8e4m3_to_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_scale,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %n, %idx;
    .reg .u64 %src, %dst, %scale_p, %addr, %off;
    .reg .b16 %packed_in, %f16lo, %f16hi_unused;
    .reg .b32 %packed_f16x2;
    .reg .f32 %vlo, %scale;
    .reg .pred %p;

    ld.param.u64 %src, [p_src];
    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %scale_p, [p_scale];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_cast_f8_end;

    ld.global.f32 %scale, [%scale_p];

    // Load one byte as low e4m3, upper byte is dummy.
    mul.wide.u32 %off, %idx, 1;
    add.u64 %addr, %src, %off;
    ld.global.b8 %packed_in, [%addr];
    // Unpack e4m3x2 -> f16x2 (lower half valid, upper half from dummy byte).
    cvt.rn.f16x2.e4m3x2 %packed_f16x2, %packed_in;
    // Extract lower f16.
    mov.b32 {%f16lo, %f16hi_unused}, %packed_f16x2;
    // Widen f16 -> f32.
    cvt.f32.f16 %vlo, %f16lo;
    mul.f32 %vlo, %vlo, %scale;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %dst, %off;
    st.global.f32 [%addr], %vlo;
$L_cast_f8_end:
    ret;
}
`
// END-OF-STAGE-5-DISABLED-BLOCK-BELOW
var _ = `
// ============================================================
// relu_f32: c = max(a, 0)
// ============================================================
.visible .entry relu_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %zero;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mov.f32 %zero, 0f00000000;
    max.f32 %vc, %va, %zero;
    st.global.f32 [%pd], %vc;
$Lrelu_f32_done:
    ret;
}

// ============================================================
// relu_f64
// ============================================================
.visible .entry relu_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f64 %va, %vc, %zero;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %va, [%pa];
    mov.f64 %zero, 0d0000000000000000;
    max.f64 %vc, %va, %zero;
    st.global.f64 [%pd], %vc;
$Lrelu_f64_done:
    ret;
}

// ============================================================
// sigmoid_f32: c = 1 / (1 + exp(-x))  via ex2.approx.f32
// ============================================================
.visible .entry sigmoid_f32(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .f32 %va, %vc, %log2e, %one, %t;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %va, [%pa];
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %one,   0f3F800000;
    neg.f32 %t, %va;
    mul.f32 %t, %t, %log2e;
    ex2.approx.f32 %t, %t;
    add.f32 %t, %t, %one;
    rcp.approx.f32 %vc, %t;
    st.global.f32 [%pd], %vc;
$Lsigmoid_f32_done:
    ret;
}

// ============================================================
// sigmoid_f64: c = 1 / (1 + exp(-x)) - exp(-x) inline fdlibm
// ============================================================
.visible .entry sigmoid_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %zero32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %twopk_bits;
    .reg .f64 %x, %negx, %invln2, %ln2H, %ln2L;
    .reg .f64 %P1, %P2, %P3, %P4, %P5, %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %c, %y, %twopk, %expmx, %result;
    .reg .s32 %k_int, %expo;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];
    neg.f64 %negx, %x;
    // Inline fdlibm exp(negx) -> %expmx
    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mov.f64 %two,    0d4000000000000000;
    mul.f64 %k_fp, %negx, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %negx, %hi;
    mul.f64 %lo, %k_fp, %ln2L;
    sub.f64 %r, %hi, %lo;
    mul.f64 %t, %r, %r;
    mul.f64 %c, %P5, %t;
    add.f64 %c, %c, %P4;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P3;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P2;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P1;
    mul.f64 %c, %c, %t;
    sub.f64 %c, %r, %c;
    sub.f64 %t, %two, %c;
    mul.f64 %r, %r, %c;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;
    add.s32 %expo, %k_int, 1023;
    shl.b32 %expo, %expo, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expo};
    mov.f64 %twopk, %twopk_bits;
    mul.f64 %expmx, %y, %twopk;
    // sigmoid = 1 / (1 + exp(-x))
    add.f64 %t, %expmx, %one;
    div.rn.f64 %result, %one, %t;
    st.global.f64 [%pd], %result;
$Lsigmoid_f64_done:
    ret;
}

// tanh_f32: hardware tanh.approx.f32 (PTX 7.0+, sm_75+).

// ============================================================
// tanh_f64: c = (exp(2x) - 1) / (exp(2x) + 1) via inline fdlibm exp
// ============================================================
.visible .entry tanh_f64(
    .param .u64 p_a,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n, %zero32;
    .reg .u64 %a, %dst, %off, %pa, %pd;
    .reg .b64 %twopk_bits;
    .reg .f64 %x, %twox, %invln2, %ln2H, %ln2L;
    .reg .f64 %P1, %P2, %P3, %P4, %P5, %one, %two;
    .reg .f64 %k_fp, %hi, %lo, %r, %t, %c, %y, %twopk, %e2x, %num, %denom, %result;
    .reg .s32 %k_int, %expo;
    .reg .pred %p;
    ld.param.u64 %a, [p_a];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %pa, %a, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %x, [%pa];
    mov.f64 %two, 0d4000000000000000;
    mul.f64 %twox, %x, %two;
    // Inline fdlibm exp(twox) -> %e2x
    mov.f64 %invln2, 0d3FF71547652B82FE;
    mov.f64 %ln2H,   0d3FE62E42FEE00000;
    mov.f64 %ln2L,   0d3DEA39EF35793C76;
    mov.f64 %P1,     0d3FC555555555553E;
    mov.f64 %P2,     0dBF66C16C16BEBD93;
    mov.f64 %P3,     0d3F11566AAF25DE2C;
    mov.f64 %P4,     0dBEBBBD41C5D26BF1;
    mov.f64 %P5,     0d3E66376972BEA4D0;
    mov.f64 %one,    0d3FF0000000000000;
    mul.f64 %k_fp, %twox, %invln2;
    cvt.rni.s32.f64 %k_int, %k_fp;
    cvt.rn.f64.s32 %k_fp, %k_int;
    mul.f64 %hi, %k_fp, %ln2H;
    sub.f64 %hi, %twox, %hi;
    mul.f64 %lo, %k_fp, %ln2L;
    sub.f64 %r, %hi, %lo;
    mul.f64 %t, %r, %r;
    mul.f64 %c, %P5, %t;
    add.f64 %c, %c, %P4;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P3;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P2;
    mul.f64 %c, %c, %t;
    add.f64 %c, %c, %P1;
    mul.f64 %c, %c, %t;
    sub.f64 %c, %r, %c;
    sub.f64 %t, %two, %c;
    mul.f64 %r, %r, %c;
    div.rn.f64 %r, %r, %t;
    sub.f64 %t, %lo, %r;
    sub.f64 %t, %t, %hi;
    sub.f64 %y, %one, %t;
    add.s32 %expo, %k_int, 1023;
    shl.b32 %expo, %expo, 20;
    mov.u32 %zero32, 0;
    mov.b64 %twopk_bits, {%zero32, %expo};
    mov.f64 %twopk, %twopk_bits;
    mul.f64 %e2x, %y, %twopk;
    // tanh = (e2x - 1) / (e2x + 1)
    sub.f64 %num, %e2x, %one;
    add.f64 %denom, %e2x, %one;
    div.rn.f64 %result, %num, %denom;
    st.global.f64 [%pd], %result;
$Ltanh_f64_done:
    ret;
}

// ============================================================
// relu_grad_f32: dX = (X > 0) ? dY : 0
// Args (via launchElementwise3): input=X, grad=dY, out=dX
// ============================================================
.visible .entry relu_grad_f32(
    .param .u64 p_x,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %x, %dy, %dst, %off, %px, %pdy, %pd;
    .reg .f32 %vx, %vdy, %vc, %zero;
    .reg .pred %pmask, %p;
    ld.param.u64 %x, [p_x];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %px, %x, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vx, [%px];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %zero, 0f00000000;
    setp.gt.f32 %pmask, %vx, %zero;
    selp.f32 %vc, %vdy, %zero, %pmask;
    st.global.f32 [%pd], %vc;
$Lrelu_grad_f32_done:
    ret;
}

// ============================================================
// relu_grad_f64
// ============================================================
.visible .entry relu_grad_f64(
    .param .u64 p_x,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %x, %dy, %dst, %off, %px, %pdy, %pd;
    .reg .f64 %vx, %vdy, %vc, %zero;
    .reg .pred %pmask, %p;
    ld.param.u64 %x, [p_x];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lrelu_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %px, %x, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vx, [%px];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %zero, 0d0000000000000000;
    setp.gt.f64 %pmask, %vx, %zero;
    selp.f64 %vc, %vdy, %zero, %pmask;
    st.global.f64 [%pd], %vc;
$Lrelu_grad_f64_done:
    ret;
}

// ============================================================
// sigmoid_grad_f32: dX = dY * Y * (1 - Y)
// Args: sigOut=Y, grad=dY, out=dX
// ============================================================
.visible .entry sigmoid_grad_f32(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f32 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vy, [%py];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %one, 0f3F800000;
    sub.f32 %t, %one, %vy;
    mul.f32 %t, %t, %vy;
    mul.f32 %vc, %vdy, %t;
    st.global.f32 [%pd], %vc;
$Lsigmoid_grad_f32_done:
    ret;
}

// ============================================================
// sigmoid_grad_f64
// ============================================================
.visible .entry sigmoid_grad_f64(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f64 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Lsigmoid_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vy, [%py];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %one, 0d3FF0000000000000;
    sub.f64 %t, %one, %vy;
    mul.f64 %t, %t, %vy;
    mul.f64 %vc, %vdy, %t;
    st.global.f64 [%pd], %vc;
$Lsigmoid_grad_f64_done:
    ret;
}

// ============================================================
// tanh_grad_f32: dX = dY * (1 - Y^2)
// Args: tanhOut=Y, grad=dY, out=dX
// ============================================================
.visible .entry tanh_grad_f32(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f32 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_grad_f32_done;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f32 %vy, [%py];
    ld.global.f32 %vdy, [%pdy];
    mov.f32 %one, 0f3F800000;
    mul.f32 %t, %vy, %vy;
    sub.f32 %t, %one, %t;
    mul.f32 %vc, %vdy, %t;
    st.global.f32 [%pd], %vc;
$Ltanh_grad_f32_done:
    ret;
}

// ============================================================
// tanh_grad_f64
// ============================================================
.visible .entry tanh_grad_f64(
    .param .u64 p_y,
    .param .u64 p_dy,
    .param .u64 p_dst,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %y, %dy, %dst, %off, %py, %pdy, %pd;
    .reg .f64 %vy, %vdy, %vc, %one, %t;
    .reg .pred %p;
    ld.param.u64 %y, [p_y];
    ld.param.u64 %dy, [p_dy];
    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $Ltanh_grad_f64_done;
    mul.wide.u32 %off, %idx, 8;
    add.u64 %py, %y, %off;
    add.u64 %pdy, %dy, %off;
    add.u64 %pd, %dst, %off;
    ld.global.f64 %vy, [%py];
    ld.global.f64 %vdy, [%pdy];
    mov.f64 %one, 0d3FF0000000000000;
    mul.f64 %t, %vy, %vy;
    sub.f64 %t, %one, %t;
    mul.f64 %vc, %vdy, %t;
    st.global.f64 [%pd], %vc;
$Ltanh_grad_f64_done:
    ret;
}
*/
`
