#include "Expressions.hpp"
#include <iostream>
#include <cassert>

#ifdef IMPLEMENT_SLOW_EVALUATION
	#include "unsupported/Eigen/KroneckerProduct"
	#include "MatrixManipulation.hpp"
	#include "FiniteDifference.hpp"
#endif

const double eps = 1e-10;

namespace TMD {

ExpressionPtr GetDerivative(const ExpressionPtr expression, unsigned int variable) {
	auto org_expr = expression->Clone();
	org_expr->MarkVariable(variable);
	if (!org_expr->_has_variable) {
		return nullptr;
	}
	auto diffed_expr = org_expr->Differentiate();
	diffed_expr->MarkDifferential();

	auto veced_expr = diffed_expr->Vectorize();
	auto result = Transpose::GetSelfType(
		veced_expr->GetTransposedDerivative()
	);


	return result;
}

ExpressionPtr Expression::GetTransposedDerivative() const {
	throw std::logic_error("should be copying instead of getting derivative");
}

void SingleOpExpression::MarkVariable(unsigned int uuid) {
	_child->MarkVariable(uuid);
	_has_variable = _child->_has_variable;
}

void SingleOpExpression::MarkDifferential() {
	_child->MarkDifferential();
	_has_differential = _child->_has_differential;
}

ExpressionPtr Negate::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Negate>(
		kNegatePriority,
		ExpressionType::kNegateOp,
		child->_rows, child->_cols,
		child->_has_variable, child->_has_differential,
		child
	);
}

void Negate::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		out << "\\left(-";
		_child->Print(out);
		out << "\\right)";
	} else {
		out << "\\left(-\\left( ";
		_child->Print(out);
		out << "\\right)\\right)";
	}
}

ExpressionPtr Negate::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Negate::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

ExpressionPtr Negate::Vectorize() const {
	return MatrixProduct::GetSelfType(
		Negate::GetSelfType(	
			IdentityMatrix::GetSelfType(_child->_rows * _child->_cols)
		),
		_child->Vectorize()
	);
	return GetSelfType(_child->Vectorize());
}

void Inverse::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		_child->Print(out);
		out << "^{-1}";
	} else {
		out << "\\left( ";
		_child->Print(out);
		out << " \\right)^{-1}";
	}
}

ExpressionPtr Inverse::GetSelfType(ExpressionPtr child) {
	assert(child->_rows == child->_cols);
	return std::make_shared<Inverse>(
		kInversePriority,
		ExpressionType::kInverseOp,
		child->_rows, child->_cols,
		child->_has_variable, child->_has_differential,
		child
	);
}

ExpressionPtr Inverse::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Inverse::Differentiate() const {
	auto inv_child1 = GetSelfType(_child->Clone());
	auto inv_child2 = GetSelfType(_child->Clone());
	auto neg_inv_child1 = Negate::GetSelfType(inv_child1);
	auto mult1 = MatrixProduct::GetSelfType(neg_inv_child1, _child->Differentiate());
	auto result = MatrixProduct::GetSelfType(mult1, inv_child2);
	return result;
}

ExpressionPtr Inverse::Vectorize() const {
	throw std::logic_error("vectorizing the inverse");
}

ExpressionPtr Determinant::GetSelfType(ExpressionPtr child) {
	assert(child->_rows == child->_cols);
	return std::make_shared<Determinant>(
		kDeterminantPriority, ExpressionType::kDeterminantOp, 1, 1, child->_has_variable, child->_has_differential, child
	);
}

void Determinant::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		out << "\\det ";
		_child->Print(out);
	} else {
		out << "\\det \\left(";
		_child->Print(out);
		out << "\\right)";
	}
}

ExpressionPtr Determinant::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Determinant::Differentiate() const {
	auto det_A = GetSelfType(_child->Clone());
	auto trans_inv_A = Transpose::GetSelfType(Inverse::GetSelfType(_child->Clone()));
	auto vec_inv_A = Vectorization::GetSelfType(trans_inv_A);
	auto trans_vec_inv_A = Transpose::GetSelfType(vec_inv_A);
	auto vec_diff_A = Vectorization::GetSelfType(_child->Differentiate());
	auto mult1 = MatrixProduct::GetSelfType(trans_vec_inv_A, vec_diff_A);
	auto result = ScalarMatrixProduct::GetSelfType(det_A, mult1);
	return result;
}

ExpressionPtr Determinant::Vectorize() const {
	throw std::logic_error("vecorizing the determinant");
}

ExpressionPtr Vectorization::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Vectorization> (
		kVectorizationPriority, ExpressionType::kVectorizationOp, child->_rows * child->_cols, 1, child->_has_variable, child->_has_differential, child
	);
}

void Vectorization::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		out << "vec";
		_child->Print(out);
	} else {
		out << "vec \\left(";
		_child->Print(out);
		out << "\\right)";
	}
}

ExpressionPtr Vectorization::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Vectorization::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

ExpressionPtr Vectorization::Vectorize() const {
	return _child->Vectorize();
}

ExpressionPtr Transpose::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Transpose>(
		kTransposePriority, ExpressionType::kTransposeOp, child->_cols, child->_rows, child->_has_variable, child->_has_differential, child
	);
}

void Transpose::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		_child->Print(out);
		out << "^T";
	} else {
		out << "\\left(";
		_child->Print(out);
		out << "\\right)^T";
	}
}

ExpressionPtr Transpose::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Transpose::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

ExpressionPtr Transpose::Vectorize() const {
	auto K = CommutationMatrix::GetSelfType(_child->_rows, _child->_cols);
	auto vec_child = _child->Vectorize();
	return MatrixProduct::GetSelfType(K, vec_child);
}

ExpressionPtr ScalarPower::GetSelfType(ExpressionPtr child, double power) {
	assert(child->_rows == 1 && child->_cols == 1);
	return std::make_shared<ScalarPower>(
		kScalarPowerPriority,
		ExpressionType::kScalarPowerOp,
		1, 1,
		child->_has_variable,
		child->_has_differential,
		child, power
	);
}

void ScalarPower::Print(std::ostream &out) const {
	if (_child->_priority > _priority) {
		_child->Print(out);
		out << "^" << _power;
	} else {
		out << "\\left(";
		_child->Print(out);
		out << "\\right)^{" << _power << "}";
	}
}

ExpressionPtr ScalarPower::Clone() const {
	return GetSelfType(_child->Clone(), _power);
}

ExpressionPtr ScalarPower::Differentiate() const {
	auto power = Constant::GetSelfType(_power);
	ExpressionPtr mult;
	if (std::abs(_power - 1) < eps) {
		mult  = _child->Differentiate();
	} else {
		auto child_power_m_1 = GetSelfType(_child->Clone(), _power - 1);
		mult = ScalarMatrixProduct::GetSelfType(child_power_m_1, _child->Differentiate());
	}
	return ScalarMatrixProduct::GetSelfType(power, mult);
}

ExpressionPtr ScalarPower::Vectorize() const {
	throw std::logic_error("vectorizing the scalar power");
}

void Diagonalization::Print(std::ostream &out) const {
	out << "diag \\left(";
	_child->Print(out);
	out << "\\right)";
}

ExpressionPtr Diagonalization::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Diagonalization::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

ExpressionPtr Diagonalization::Vectorize() const {
	return MatrixProduct::GetSelfType(
		DiagonalizationMatrix::GetSelfType(_child->_rows),
		_child->Vectorize()
	);
}

ExpressionPtr Diagonalization::GetSelfType(ExpressionPtr child) {
	assert(child->_cols == 1);
	return std::make_shared<Diagonalization>(
		kDiagonalizePriority, ExpressionType::kDiagonalizeOp,
		child->_rows, child->_rows,
		child->_has_variable, child->_has_differential,
		child
	);
}

void Exp::Print(std::ostream &out) const {
	out << "\\exp \\left(";
	_child->Print(out);
	out << "\\right)";
}

ExpressionPtr Exp::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Exp::Differentiate() const {
	return HadamardProduct::GetSelfType(
		GetSelfType(_child->Clone()),
		_child->Differentiate()
	);
}

ExpressionPtr Exp::Vectorize() const {
	throw std::logic_error("vectorizing an exp expression");
}

ExpressionPtr Exp::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Exp>(kExpPriority, ExpressionType::kExpOp, child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

void DoubleOpExpression::MarkVariable(unsigned int uuid) {
	_lhs->MarkVariable(uuid);
	_rhs->MarkVariable(uuid);
	_has_variable = (_lhs->_has_variable || _rhs->_has_variable);
}

void DoubleOpExpression::MarkDifferential() {
	_lhs->MarkDifferential();
	_rhs->MarkDifferential();
	_has_differential = (_lhs->_has_differential || _rhs->_has_differential);
}

void DoubleOpExpression::Print(std::ostream &out) const {
	if (_lhs->_priority >= _priority) {
		_lhs->Print(out);
	} else {
		out << "\\left(";
		_lhs->Print(out);
		out << "\\right)";
	}
	if (_operator_sign.length() > 0) {
		out << " " << _operator_sign << " ";
	}
	if (_rhs->_priority > _priority) {
		_rhs->Print(out);
	} else if (_rhs->_type == _type) {
		// TODO: consider left/right composition order
		_rhs->Print(out);
	} else {
		out << "\\left(";
		_rhs->Print(out);
		out << "\\right)";
	}
}

ExpressionPtr MatrixAddition::GetTransposedDerivative() const {
	if (!_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("getting derivative on expression that has no differential");
	} else if (_lhs->_has_differential && !_rhs->_has_differential) {
		return _lhs->GetTransposedDerivative();
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return _rhs->GetTransposedDerivative();
	} else {
		return GetSelfType(_lhs->GetTransposedDerivative(), _rhs->GetTransposedDerivative());
	}
}

ExpressionPtr MatrixAddition::GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs) {
	if (lhs->_rows != rhs->_rows || lhs->_cols != rhs->_cols) {
		throw std::logic_error("adding matrix of different shape");
	}
	return std::make_shared<MatrixAddition>(
		kMatrixAdditionPriority,
		ExpressionType::kMatrixAdditionOp,
		"+",
		lhs->_rows, lhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

ExpressionPtr MatrixAddition::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr MatrixAddition::Differentiate() const {
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return GetSelfType(_lhs->Differentiate(), _rhs->Differentiate());
	} else if (_lhs->_has_variable && !_rhs->_has_variable) {
		return _lhs->Differentiate();
	} else if (!_lhs->_has_variable && _rhs->_has_variable) {
		return _rhs->Differentiate();
	} else {
		throw std::logic_error("differentiating an expression without variable");
	}
}

ExpressionPtr MatrixAddition::Vectorize() const {
	if (_lhs->_has_differential && _rhs->_has_differential) {
		return GetSelfType(_lhs->Vectorize(), _rhs->Vectorize());
	} else {
		throw std::logic_error("differential appears only on one side of addition");
	}
}

ExpressionPtr MatrixProduct::GetTransposedDerivative() const {
	if (!_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("getting derivatives on expressions without differential");
	} else if (_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("differential appears on lhs of matrix product");
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return GetSelfType(_lhs->Clone(), _rhs->GetTransposedDerivative());
	} else {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
}

ExpressionPtr MatrixProduct::GetSelfType(
	ExpressionPtr lhs, ExpressionPtr rhs
) {
	if (lhs->_cols != rhs->_rows) {
		throw std::logic_error("the shape of matrices in product does not match");
	}
	return std::make_shared<MatrixProduct>(
		kMatrixProductPriority,
		ExpressionType::kMatrixProductOp,
		"",
		lhs->_rows, rhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

ExpressionPtr MatrixProduct::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr MatrixProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->Differentiate(), _rhs->Clone()),
			GetSelfType(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a matrix product with no varaible");
}

ExpressionPtr MatrixProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto BT = Transpose::GetSelfType(_rhs->Clone());
		auto BTKronI = KroneckerProduct::GetSelfType(
			BT, IdentityMatrix::GetSelfType(_lhs->_rows)
		);
		return MatrixProduct::GetSelfType(
			BTKronI, _lhs->Vectorize()
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto IKronA = KroneckerProduct::GetSelfType(
			IdentityMatrix::GetSelfType(_rhs->_cols), _lhs->Clone()
		);
		return MatrixProduct::GetSelfType(
			IKronA, _rhs->Vectorize()
		);
	}
	throw std::logic_error("vectorizing a matrix product without differentials");
}

ExpressionPtr KroneckerProduct::GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs) {
	return std::make_shared<KroneckerProduct>(
		kKroneckerProductPriority,
		ExpressionType::kKroneckerProductOp,
		"\\otimes",
		lhs->_rows * rhs->_rows,
		lhs->_cols * rhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}


ExpressionPtr KroneckerProduct::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr KroneckerProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->Differentiate(), _rhs->Clone()),
			GetSelfType(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a Kronecker product with no varaible");
}

ExpressionPtr KroneckerProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto I_kron_K_kron_I = GetSelfType(
			IdentityMatrix::GetSelfType(_lhs->_cols),
			GetSelfType(
				CommutationMatrix::GetSelfType(_rhs->_cols, _lhs->_rows),
				IdentityMatrix::GetSelfType(_rhs->_rows)
			)
		);
		auto vec_A_kron_I = GetSelfType(
			Vectorization::GetSelfType(_lhs->Clone()),
			IdentityMatrix::GetSelfType(_rhs->_rows * _rhs->_cols)
		);
		return MatrixProduct::GetSelfType(
			I_kron_K_kron_I,
			MatrixProduct::GetSelfType(
				vec_A_kron_I, _rhs->Vectorize()
			)
		);
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto K_kron_K = GetSelfType(
			CommutationMatrix::GetSelfType(_lhs->_cols, _rhs->_cols),
			CommutationMatrix::GetSelfType(_lhs->_rows, _rhs->_rows)
		);
		auto I_kron_K_kron_I = GetSelfType(
			IdentityMatrix::GetSelfType(_rhs->_cols),
			GetSelfType(
				CommutationMatrix::GetSelfType(_lhs->_cols, _rhs->_rows),
				IdentityMatrix::GetSelfType(_lhs->_rows)
			)
		);
		auto vec_B_kron_I = GetSelfType(
			Vectorization::GetSelfType(_rhs->Clone()),
			IdentityMatrix::GetSelfType(_lhs->_rows * _lhs->_cols)
		);
		return MatrixProduct::GetSelfType(
			K_kron_K,
			MatrixProduct::GetSelfType(
				I_kron_K_kron_I,
				MatrixProduct::GetSelfType(
					vec_B_kron_I, _lhs->Vectorize()
				)
			)
		);
	}
	throw std::logic_error("vectorizing a Kronecker product without differentials");
}

ExpressionPtr ScalarMatrixProduct::GetSelfType(
	ExpressionPtr scalar, ExpressionPtr matrix
) {
	if (scalar->_rows != 1 || scalar->_cols != 1) {
		throw std::logic_error("scalar-matrix product with non-scalar");
	}
	return std::make_shared<ScalarMatrixProduct>(
		kScalarMatrixProductPriority,
		ExpressionType::kScalarMatrixProductOp,
		"*",
		matrix->_rows, matrix->_cols,
		scalar->_has_variable || matrix->_has_variable,
		scalar->_has_differential || matrix->_has_differential,
		scalar, matrix
	);
}

ExpressionPtr ScalarMatrixProduct::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr ScalarMatrixProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->Differentiate(), _rhs->Clone()),
			GetSelfType(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a scalar-matrix product with no varaible");
}

ExpressionPtr ScalarMatrixProduct::Vectorize() const {
	// a * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of scalar-matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto vec_B = Vectorization::GetSelfType(_rhs->Clone());
		return MatrixProduct::GetSelfType(
			vec_B, _lhs->Vectorize()
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			KroneckerProduct::GetSelfType(_lhs->Clone(), IdentityMatrix::GetSelfType(_rhs->_rows * _rhs->_cols)),
			_rhs->Vectorize()
		);
	}
	throw std::logic_error("vectorizing a scalar-matrix product without differentials");
}

ExpressionPtr HadamardProduct::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr HadamardProduct::Differentiate() const {
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			HadamardProduct::GetSelfType(_rhs->Clone(), _lhs->Differentiate()),
			HadamardProduct::GetSelfType(_lhs->Clone(), _rhs->Differentiate())
		);
	} else if (_lhs->_has_variable && !_rhs->_has_variable) {
		return HadamardProduct::GetSelfType(_rhs->Clone(), _lhs->Differentiate());
	} else if (!_lhs->_has_variable && _rhs->_has_variable) {
		return HadamardProduct::GetSelfType(_lhs->Clone(), _rhs->Differentiate());
	} else {
		throw std::logic_error("differentiating a hadamard product with no variable");
	}
}

ExpressionPtr HadamardProduct::Vectorize() const {
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			Diagonalization::GetSelfType(
				Vectorization::GetSelfType(
					_rhs->Clone()
				)
			),
			_lhs->Vectorize()
		);
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			Diagonalization::GetSelfType(
				Vectorization::GetSelfType(
					_lhs->Clone()
				)
			),
			_rhs->Vectorize()
		);
	} else if (!_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("vectorizing a hadamard product without differential");
	} else {
		throw std::logic_error("differential appears on both sides of hadarmard product");
	}
}

ExpressionPtr HadamardProduct::GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs) {
	assert(lhs->_rows == rhs->_rows && lhs->_cols == rhs->_cols);
	return std::make_shared<HadamardProduct> (
		kHadamardProductPriority,
		ExpressionType::kHadamardProductOp,
		"\\odot",
		lhs->_rows, lhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

void LeafExpression::MarkVariable(unsigned int uuid) {
	_has_variable = false;
}

void LeafExpression::MarkDifferential() {}

ExpressionPtr IdentityMatrix::GetSelfType(int order) {
	return std::make_shared<IdentityMatrix>(order, order, false, false, order);
}

void IdentityMatrix::Print(std::ostream &out) const {
	out << "I_{" << _order << "}";
}

ExpressionPtr IdentityMatrix::Clone() const {
	return GetSelfType(_order);
}

ExpressionPtr IdentityMatrix::Differentiate() const {
	throw std::logic_error("differentiating an identity matrix");
}

ExpressionPtr IdentityMatrix::Vectorize() const {
	throw std::logic_error("vectorizing an identity matrix");
}

ExpressionPtr CommutationMatrix::GetSelfType(int m, int n) {
	if (m == 1 || n == 1) {
		return IdentityMatrix::GetSelfType(m * n);
	} else {
		return std::make_shared<CommutationMatrix>(
			m * n, m * n,
			false, false, m, n
		);
	}
}

void CommutationMatrix::Print(std::ostream &out) const {
	out << "K_{" << _m << ", " << _n << "}";
}

ExpressionPtr CommutationMatrix::Clone() const {
	return GetSelfType(_m, _n);
}

ExpressionPtr CommutationMatrix::Differentiate() const {
	throw std::logic_error("differentiating a commutation matrix");
}

ExpressionPtr CommutationMatrix::Vectorize() const {
	throw std::logic_error("vectorizing a commutation matrix");
}

ExpressionPtr DiagonalizationMatrix::GetSelfType(int order) {
	return std::make_shared<DiagonalizationMatrix>(order * order, order, false, false, order);
}

void DiagonalizationMatrix::Print(std::ostream &out) const {
	out << "D_{" << _order << "}";
}

ExpressionPtr DiagonalizationMatrix::Clone() const {
	return GetSelfType(_order);
}

ExpressionPtr DiagonalizationMatrix::Differentiate() const {
	throw std::logic_error("differenticating diagonalization matrix");
}

ExpressionPtr DiagonalizationMatrix::Vectorize() const {
	throw std::logic_error("vectorizing diagonalization matrix");
}

ExpressionPtr Constant::GetSelfType(double value) {
	return std::make_shared<Constant>(1, 1, false, false, value);
}

void Constant::Print(std::ostream &out) const {
	out << _value;
}

ExpressionPtr Constant::Clone() const {
	return Constant::GetSelfType(_value);
}

ExpressionPtr Constant::Differentiate() const {
	throw std::logic_error("differentiating a scalar constant");
}

ExpressionPtr Constant::Vectorize() const {
	throw std::logic_error("vectorizing a scalar constant");
}

void Variable::MarkVariable(unsigned int uuid) {
	_has_variable = (uuid == _uuid);
}

ExpressionPtr Variable::GetTransposedDerivative() const {
	if (_has_differential) {
		return IdentityMatrix::GetSelfType(_rows);
	} else {
		throw std::logic_error("getting derivatives on expressions without differential");
	}
}

std::shared_ptr<Variable> Variable::GetSelfType(const std::string &name, unsigned int uuid, int rows, int cols) {
	return std::make_shared<Variable>(rows, cols, false, false, name, uuid);
}

void Variable::Print(std::ostream &out) const {
	if (_has_differential) {
		out << "d";
	}
	out << _name;
}

ExpressionPtr Variable::Clone() const {
	return std::make_shared<Variable>(_rows, _cols, _has_variable, _has_differential, _name, _uuid);
}

ExpressionPtr Variable::Differentiate() const {
	auto new_var = GetSelfType(_name, _uuid, _rows, _cols);
	new_var->_has_differential = true;
	return new_var;
}

ExpressionPtr Variable::Vectorize() const {
	return std::make_shared<Variable>(_rows * _cols, 1, _has_variable, _has_differential, _name, _uuid);
}

std::ostream& operator<<(std::ostream& out, const Expression& expr) {
	expr.Print(out);
	return out;
}

unsigned int UUIDGenerator::_cnt = 0;

unsigned int UUIDGenerator::GenUUID() {
	return _cnt++;
}


#ifdef IMPLEMENT_SLOW_EVALUATION

Eigen::MatrixXd Negate::SlowEvaluation(const VariableTable& table) const {
	return -_child->SlowEvaluation(table);
}

Eigen::MatrixXd Inverse::SlowEvaluation(const VariableTable& table) const {
	return _child->SlowEvaluation(table).inverse();
}

Eigen::MatrixXd Determinant::SlowEvaluation(const VariableTable& table) const {
	return (Eigen::MatrixXd(1, 1) << _child->SlowEvaluation(table).determinant()).finished();
}

Eigen::MatrixXd Vectorization::SlowEvaluation(const VariableTable& table) const {
	Eigen::MatrixXd unveced_mat = _child->SlowEvaluation(table);
	
	int rows = unveced_mat.rows(), cols = unveced_mat.cols();
	Eigen::MatrixXd result(unveced_mat.rows() * unveced_mat.cols(), 1);

	for (int i = 0; i < cols; i++) {
		result.block(i * rows, 0, rows, 1) = unveced_mat.col(i);
	}
	return result;
}

Eigen::MatrixXd Transpose::SlowEvaluation(const VariableTable& table) const {
	return _child->SlowEvaluation(table).transpose();
}

Eigen::MatrixXd ScalarPower::SlowEvaluation(const VariableTable& table) const {
	auto child_result = _child->SlowEvaluation(table);
	return (Eigen::MatrixXd(1, 1) << std::pow(child_result(0, 0), _power)).finished();
}

Eigen::MatrixXd Diagonalization::SlowEvaluation(const VariableTable &table) const {
	return _child->SlowEvaluation(table).asDiagonal();
}

Eigen::MatrixXd Exp::SlowEvaluation(const VariableTable &table) const {
	return _child->SlowEvaluation(table).array().exp();
}

Eigen::MatrixXd MatrixAddition::SlowEvaluation(const VariableTable& table) const {
	return _lhs->SlowEvaluation(table) + _rhs->SlowEvaluation(table);
}

Eigen::MatrixXd MatrixProduct::SlowEvaluation(const VariableTable& table) const {
	return _lhs->SlowEvaluation(table) * _rhs->SlowEvaluation(table);
}

Eigen::MatrixXd KroneckerProduct::SlowEvaluation(const VariableTable& table) const {
	return Eigen::kroneckerProduct(
		_lhs->SlowEvaluation(table),
		_rhs->SlowEvaluation(table)
	);
}

Eigen::MatrixXd ScalarMatrixProduct::SlowEvaluation(const VariableTable& table) const {
	return _lhs->SlowEvaluation(table)(0, 0) * _rhs->SlowEvaluation(table);
}

Eigen::MatrixXd HadamardProduct::SlowEvaluation(const VariableTable &table) const {
	return _lhs->SlowEvaluation(table).cwiseProduct(_rhs->SlowEvaluation(table));
}

Eigen::MatrixXd IdentityMatrix::SlowEvaluation(const VariableTable& table) const {
	return Eigen::MatrixXd::Identity(_order, _order);
}

Eigen::MatrixXd CommutationMatrix::SlowEvaluation(const VariableTable& table) const {
	return Numerics::GetPermutationMatrix(_m, _n);
}

Eigen::MatrixXd DiagonalizationMatrix::SlowEvaluation(const VariableTable &table) const {
	Eigen::MatrixXd D = Eigen::MatrixXd::Zero(_rows, _cols);
	for (int i = 0; i < _order; i++) {
		D(i * _order + i, i) = 1;
	}
	return D;
}

Eigen::MatrixXd Constant::SlowEvaluation(const VariableTable& table) const {
	return (Eigen::MatrixXd(1, 1) << _value).finished();
}

Eigen::MatrixXd Variable::SlowEvaluation(const VariableTable& table) const {
	auto itr = table.find(_uuid);
	if (itr == table.end()) {
		throw std::logic_error("cannot find variable");
	} else {
		if (itr->second.rows() != _rows || itr->second.cols() != _cols) {
			throw std::logic_error("variable value does not match specified size");
		}
		return itr->second;
	}
}

Eigen::MatrixXd GetExpressionNumericDerivative(
	const TMD::ExpressionPtr expression,
	const TMD::VariableTable& table,
	unsigned int uuid
) {
	TMD::VariableTable tmp_table = table;
	auto func = [&tmp_table, &expression, uuid] (const Eigen::MatrixXd X) -> Eigen::MatrixXd {
		tmp_table[uuid] = X;
		return expression->SlowEvaluation(tmp_table);
	};
	return Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func, table.at(uuid), 1e-4);
}

#endif

}