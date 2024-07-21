use stwo_prover::core::{
    backend::{Col, Column},
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
};

use crate::{
    backend::CudaBackend,
    cuda::{self},
};

impl PolyOps for CudaBackend {
    type Twiddles = cuda::BaseFieldVec;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let size = values.len();
        let device_ptr = unsafe {
            cuda::bindings::sort_values_and_permute_with_bit_reverse_order(values.device_ptr, size)
        };
        let result = cuda::BaseFieldVec::new(device_ptr, size);
        CircleEvaluation::new(coset.circle_domain(), result)
    }

    fn interpolate(
        _eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        _itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!()
    }

    fn eval_at_point(_poly: &CirclePoly<Self>, _point: CirclePoint<SecureField>) -> SecureField {
        todo!()
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size"
        );

        let mut new_coeffs = cuda::BaseFieldVec::new_zeroes(new_size);
        new_coeffs.copy_from(&poly.coeffs);
        CirclePoly::new(new_coeffs)
    }

    fn evaluate(
        _poly: &CirclePoly<Self>,
        _domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles = cuda::BaseFieldVec::new(
                cuda::bindings::precompute_twiddles(
                    coset.initial.into(),
                    coset.step.into(),
                    coset.size(),
                ),
                coset.size(),
            );
            let itwiddles = cuda::BaseFieldVec::new_uninitialized(coset.size());
            cuda::bindings::batch_inverse_base_field(
                twiddles.device_ptr,
                itwiddles.device_ptr,
                coset.size(),
            );
            TwiddleTree {
                root_coset: coset,
                twiddles,
                itwiddles,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::CudaBackend, cuda};
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        fields::m31::BaseField,
        poly::circle::{CanonicCoset, CirclePoly, PolyOps},
    };

    #[test]
    fn test_new_canonical_ordered() {
        let log_size = 25;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset, cpu_values);

        let column = cuda::BaseFieldVec::from_vec(column_data);
        let result = CudaBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
        assert_eq!(
            result.domain.iter().collect::<Vec<_>>(),
            expected_result.domain.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 20;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset);
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extend() {
        let log_size = 20;
        let size = 1 << log_size;
        let new_log_size = log_size + 5;
        let cpu_coeffs = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let cuda_coeffs = cuda::BaseFieldVec::from_vec(cpu_coeffs.clone());
        let cpu_poly = CirclePoly::<CpuBackend>::new(cpu_coeffs);
        let cuda_poly = CirclePoly::<CudaBackend>::new(cuda_coeffs);
        let result = CudaBackend::extend(&cuda_poly, new_log_size);
        let expected_result = CpuBackend::extend(&cpu_poly, new_log_size);
        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
        assert_eq!(result.log_size(), expected_result.log_size());
    }
}
