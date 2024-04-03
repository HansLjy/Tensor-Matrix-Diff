#include "Expressions.hpp"
#include <iostream>
#include <cassert>

const double eps = 1e-10;

namespace TMD {

Expression* GetDerivative(const Expression* expression, unsigned int variable) {
	auto org_expr = expression->Clone();
	org_expr->Print(std::cerr);
	std::cerr << std::endl;
	org_expr->MarkVariable(variable);
	auto diffed_expr = org_expr->Differentiate();
	diffed_expr->MarkDifferential();
	diffed_expr->Print(std::cerr);
	std::cerr << std::endl;
	auto veced_expr = diffed_expr->Vectorize();
	veced_expr->Print(std::cerr);
	std::cerr << std::endl;
	auto result = veced_expr->GetDerivative();
	result->Print(std::cerr);
	std::cerr << std::endl;

	delete org_expr;
	delete diffed_expr;
	delete veced_expr;
	return result;
}

Expression* Expression::GetDerivative() const {
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

SingleOpExpression::~SingleOpExpression() {
	delete _child;
}

ExpressionType Negate::GetExpressionType() const {
	return ExpressionType::kNegateOp;
}

Negate* Negate::GetNegateExpression(Expression *child) {
	return new Negate(child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

void Negate::Print(std::ostream &out) const {
	out << "-\\left\\( ";
	_child->Print(out);
	out << "\\right\\)";
}

Expression* Negate::Clone() const {
	return GetNegateExpression(_child->Clone());
}

Expression* Negate::Differentiate() const {
	return GetNegateExpression(_child->Differentiate());
}

Expression* Negate::Vectorize() const {
	return GetNegateExpression(_child->Vectorize());
}

ExpressionType Inverse::GetExpressionType() const {
	return ExpressionType::kInverseOp;
}

void Inverse::Print(std::ostream &out) const {
	out << "\\left\\( ";
	_child->Print(out);
	out << " \\right\\)^{-1}";
}

Inverse* Inverse::GetInverseExpression(Expression *child) {
	assert(child->_rows == child->_cols);
	return new Inverse(child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

Expression* Inverse::Clone() const {
	return GetInverseExpression(_child->Clone());
}

Expression* Inverse::Differentiate() const {
	auto inv_child1 = GetInverseExpression(_child->Clone());
	auto inv_child2 = GetInverseExpression(_child->Clone());
	auto neg_inv_child1 = Negate::GetNegateExpression(inv_child1);
	auto mult1 = MatrixProduct::GetMatrixProductExpression(neg_inv_child1, _child->Differentiate());
	auto result = MatrixProduct::GetMatrixProductExpression(mult1, inv_child2);
	return result;
}

Expression* Inverse::Vectorize() const {
	throw std::logic_error("vectorizing the inverse");
}

ExpressionType Determinant::GetExpressionType() const {
	return ExpressionType::kDeterminantOp;
}

Determinant* Determinant::GetMatrixDeterminantExpression(Expression *child) {
	assert(child->_rows == child->_cols);
	return new Determinant(
		1, 1, child->_has_variable, child->_has_differential, child
	);
}

void Determinant::Print(std::ostream &out) const {
	out << "\\det \\left(";
	_child->Print(out);
	out << "\\right)";
}

Expression * Determinant::Clone() const {
	return GetMatrixDeterminantExpression(_child->Clone());
}

Expression * Determinant::Differentiate() const {
	auto det_A = GetMatrixDeterminantExpression(_child->Clone());
	auto inv_A = Inverse::GetInverseExpression(_child->Clone());
	auto vec_inv_A = Vectorization::GetVectorizationExpression(inv_A);
	auto trans_vec_inv_A = Transpose::GetTransposeExpression(vec_inv_A);
	auto vec_diff_A = Vectorization::GetVectorizationExpression(_child->Differentiate());
	auto mult1 = MatrixProduct::GetMatrixProductExpression(trans_vec_inv_A, vec_diff_A);
	auto result = ScalarMatrixProduct::GetScalarMatrixProduct(det_A, mult1);
	return result;
}

Expression * Determinant::Vectorize() const {
	throw std::logic_error("vecorizing the determinant");
}

ExpressionType Vectorization::GetExpressionType() const {
	return ExpressionType::kVectorizationOp;
}

Vectorization* Vectorization::GetVectorizationExpression(Expression *child) {
	return new Vectorization(
		child->_rows * child->_cols, 1, child->_has_variable, child->_has_differential, child
	);
}

void Vectorization::Print(std::ostream &out) const {
	out << "vec \\left(";
	_child->Print(out);
	out << "\\right)";
}

Expression* Vectorization::Clone() const {
	return GetVectorizationExpression(_child->Clone());
}

Expression* Vectorization::Differentiate() const {
	return GetVectorizationExpression(_child->Differentiate());
}

Expression* Vectorization::Vectorize() const {
	return _child->Vectorize();
}

ExpressionType Transpose::GetExpressionType() const {
	return ExpressionType::kTransposeOp;
}

Transpose* Transpose::GetTransposeExpression(Expression *child) {
	return new Transpose(
		child->_cols, child->_rows, child->_has_variable, child->_has_differential, child
	);
}

void Transpose::Print(std::ostream &out) const {
	out << "\\left(";
	_child->Print(out);
	out << "\\right)^T";
}

Expression * Transpose::Clone() const {
	return GetTransposeExpression(_child->Clone());
}

Expression * Transpose::Differentiate() const {
	return GetTransposeExpression(_child->Differentiate());
}

Expression * Transpose::Vectorize() const {
	auto K = CommutationMatrix::GetCommutationMatrix(_child->_rows, _child->_cols);
	auto vec_child = _child->Vectorize();
	return MatrixProduct::GetMatrixProductExpression(K, vec_child);
}

ExpressionType ScalarPower::GetExpressionType() const {
	return ExpressionType::kScalarPowerOp;
}

ScalarPower* ScalarPower::GetScalarPowerExpression(Expression *child, double power) {
	assert(child->_rows == 1 && child->_cols == 1);
	return new ScalarPower(
		1, 1,
		child->_has_variable,
		child->_has_differential,
		child, power
	);
}

void ScalarPower::Print(std::ostream &out) const {
	out << "\\left(";
	_child->Print(out);
	out << "\\right)^" << _power;
}

Expression * ScalarPower::Clone() const {
	return GetScalarPowerExpression(_child->Clone(), _power);
}

Expression * ScalarPower::Differentiate() const {
	auto power = ScalarConstant::GetScalarConstant(_power);
	Expression* mult;
	if (std::abs(_power - 1) < eps) {
		mult  = _child->Differentiate();
	} else {
		auto child_power_m_1 = GetScalarPowerExpression(_child->Clone(), _power - 1);
		mult = ScalarMatrixProduct::GetScalarMatrixProduct(child_power_m_1, _child->Differentiate());
	}
	return ScalarMatrixProduct::GetScalarMatrixProduct(power, mult);
}

Expression * ScalarPower::Vectorize() const {
	throw std::logic_error("vectorizing the scalar power");
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

DoubleOpExpression::~DoubleOpExpression() {
	delete _lhs;
	delete _rhs;
}

ExpressionType MatrixAddition::GetExpressionType() const {
	return ExpressionType::kMatrixAdditionOp;
}

Expression* MatrixAddition::GetDerivative() const {
	if (!_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("getting derivative on expression that has no differential");
	} else if (_lhs->_has_differential && !_rhs->_has_differential) {
		return _lhs->GetDerivative();
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return _rhs->GetDerivative();
	} else {
		return GetMatrixAdditionExpression(_lhs->GetDerivative(), _rhs->GetDerivative());
	}
}

MatrixAddition* MatrixAddition::GetMatrixAdditionExpression(Expression *lhs, Expression *rhs) {
	if (lhs->_rows != rhs->_rows || lhs->_cols != rhs->_cols) {
		throw std::logic_error("adding matrix of different shape");
	}
	return new MatrixAddition(
		lhs->_rows, lhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

void MatrixAddition::Print(std::ostream &out) const {
	out << "\\left(";
	_lhs->Print(out);
	out << "\\right) + \\left(";
	_rhs->Print(out);
	out << "\\right)";
}

Expression* MatrixAddition::Clone() const {
	return GetMatrixAdditionExpression(_lhs->Clone(), _rhs->Clone());
}

Expression* MatrixAddition::Differentiate() const {
	return GetMatrixAdditionExpression(_lhs->Differentiate(), _rhs->Differentiate());
}

Expression* MatrixAddition::Vectorize() const {
	return GetMatrixAdditionExpression(_lhs->Vectorize(), _rhs->Vectorize());
}

ExpressionType MatrixProduct::GetExpressionType() const {
	return ExpressionType::kMatrixProductOp;
}

Expression* MatrixProduct::GetDerivative() const {
	if (!_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("getting derivatives on expressions without differential");
	} else if (_lhs->_has_differential && !_rhs->_has_differential) {
		throw std::logic_error("differential appears on lhs of matrix product");
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return GetMatrixProductExpression(_lhs->Clone(), _rhs->GetDerivative());
	} else {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
}

MatrixProduct* MatrixProduct::GetMatrixProductExpression(
	Expression *lhs, Expression *rhs
) {
	if (lhs->_cols != rhs->_rows) {
		throw std::logic_error("the shape of matrices in product does not match");
	}
	return new MatrixProduct(
		lhs->_rows, rhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

void MatrixProduct::Print(std::ostream &out) const {
	out << "\\left(";
	_lhs->Print(out);
	out << "\\right) \\left(";
	_rhs->Print(out);
	out << "\\right)";
}

Expression * MatrixProduct::Clone() const {
	return GetMatrixProductExpression(_lhs->Clone(), _rhs->Clone());
}

Expression * MatrixProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetMatrixProductExpression(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetMatrixProductExpression(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetMatrixAdditionExpression(
			GetMatrixProductExpression(_lhs->Differentiate(), _rhs->Clone()),
			GetMatrixProductExpression(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a matrix product with no varaible");
}

Expression * MatrixProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto BT = Transpose::GetTransposeExpression(_rhs->Clone());
		auto BTKronI = KroneckerProduct::GetKroneckerProduct(
			BT, IdentityMatrix::GetIdentityMatrix(_lhs->_rows)
		);
		return MatrixProduct::GetMatrixProductExpression(
			BTKronI, _lhs->Vectorize()
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto IKronA = KroneckerProduct::GetKroneckerProduct(
			IdentityMatrix::GetIdentityMatrix(_rhs->_cols), _lhs->Clone()
		);
		return MatrixProduct::GetMatrixProductExpression(
			IKronA, _rhs->Vectorize()
		);
	}
	throw std::logic_error("vectorizing a matrix product without differentials");
}

ExpressionType KroneckerProduct::GetExpressionType() const {
	return ExpressionType::kKroneckerProductOp;
}

KroneckerProduct* KroneckerProduct::GetKroneckerProduct(Expression *lhs, Expression *rhs) {
	return new KroneckerProduct(
		lhs->_rows * rhs->_rows,
		lhs->_cols * rhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

void KroneckerProduct::Print(std::ostream &out) const {
	out << "\\left(";
	_lhs->Print(out);
	out << "\\right) \\otimes \\left(";
	_rhs->Print(out);
	out << "\\right)";
}

Expression * KroneckerProduct::Clone() const {
	return GetKroneckerProduct(_lhs->Clone(), _rhs->Clone());
}

Expression * KroneckerProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetKroneckerProduct(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetKroneckerProduct(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetMatrixAdditionExpression(
			GetKroneckerProduct(_lhs->Differentiate(), _rhs->Clone()),
			GetKroneckerProduct(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a Kronecker product with no varaible");
}

Expression * KroneckerProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto I_kron_K_kron_I = GetKroneckerProduct(
			IdentityMatrix::GetIdentityMatrix(_lhs->_cols),
			GetKroneckerProduct(
				CommutationMatrix::GetCommutationMatrix(_rhs->_cols, _lhs->_rows),
				IdentityMatrix::GetIdentityMatrix(_rhs->_rows)
			)
		);
		auto vec_A_kron_I = GetKroneckerProduct(
			Vectorization::GetVectorizationExpression(_lhs->Clone()),
			IdentityMatrix::GetIdentityMatrix(_rhs->_rows * _rhs->_cols)
		);
		return MatrixProduct::GetMatrixProductExpression(
			I_kron_K_kron_I,
			MatrixProduct::GetMatrixProductExpression(
				vec_A_kron_I, _rhs->Vectorize()
			)
		);
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto K_kron_K = GetKroneckerProduct(
			CommutationMatrix::GetCommutationMatrix(_lhs->_cols, _rhs->_cols),
			CommutationMatrix::GetCommutationMatrix(_lhs->_rows, _rhs->_rows)
		);
		auto I_kron_K_kron_I = GetKroneckerProduct(
			IdentityMatrix::GetIdentityMatrix(_rhs->_cols),
			GetKroneckerProduct(
				CommutationMatrix::GetCommutationMatrix(_lhs->_cols, _rhs->_rows),
				IdentityMatrix::GetIdentityMatrix(_lhs->_rows)
			)
		);
		auto vec_B_kron_I = GetKroneckerProduct(
			Vectorization::GetVectorizationExpression(_rhs->Clone()),
			IdentityMatrix::GetIdentityMatrix(_lhs->_rows * _lhs->_cols)
		);
		return MatrixProduct::GetMatrixProductExpression(
			K_kron_K,
			MatrixProduct::GetMatrixProductExpression(
				I_kron_K_kron_I,
				MatrixProduct::GetMatrixProductExpression(
					vec_B_kron_I, _lhs->Vectorize()
				)
			)
		);
	}
	throw std::logic_error("vectorizing a Kronecker product without differentials");
}

ExpressionType ScalarMatrixProduct::GetExpressionType() const {
	return ExpressionType::kScalarMatrixProductOp;
}

ScalarMatrixProduct* ScalarMatrixProduct::GetScalarMatrixProduct(
	Expression *scalar, Expression *matrix
) {
	if (scalar->_rows != 1 || scalar->_cols != 1) {
		throw std::logic_error("scalar-matrix product with non-scalar");
	}
	return new ScalarMatrixProduct(
		matrix->_rows, matrix->_cols,
		scalar->_has_variable || matrix->_has_variable,
		scalar->_has_differential || matrix->_has_differential,
		scalar, matrix
	);
}

void ScalarMatrixProduct::Print(std::ostream &out) const {
	out << "\\left(";
	_lhs->Print(out);
	out << "\\right)\\left(";
	_rhs->Print(out);
	out << "\\right)";
}

Expression * ScalarMatrixProduct::Clone() const {
	return GetScalarMatrixProduct(_lhs->Clone(), _rhs->Clone());
}

Expression * ScalarMatrixProduct::Differentiate() const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetScalarMatrixProduct(_lhs->Differentiate(), _rhs->Clone());
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetScalarMatrixProduct(_lhs->Clone(), _rhs->Differentiate());
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetMatrixAdditionExpression(
			GetScalarMatrixProduct(_lhs->Differentiate(), _rhs->Clone()),
			GetScalarMatrixProduct(_lhs->Clone(), _rhs->Differentiate())
		);
	}
	throw std::logic_error("differentiating a scalar-matrix product with no varaible");
}

Expression * ScalarMatrixProduct::Vectorize() const {
	// a * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of scalar-matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto vec_B = Vectorization::GetVectorizationExpression(_rhs->Clone());
		return MatrixProduct::GetMatrixProductExpression(
			vec_B, _lhs->Vectorize()
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		return MatrixProduct::GetMatrixProductExpression(
			_rhs->Vectorize(), _lhs->Clone()
		);
	}
	throw std::logic_error("vectorizing a scalar-matrix product without differentials");
}


void LeafExpression::MarkVariable(unsigned int uuid) {
	_has_variable = false;
}

void LeafExpression::MarkDifferential() {}

ExpressionType IdentityMatrix::GetExpressionType() const {
	return ExpressionType::kIdentityMatrix;
}

IdentityMatrix* IdentityMatrix::GetIdentityMatrix(int order) {
	return new IdentityMatrix(order, order, false, false, order);
}

void IdentityMatrix::Print(std::ostream &out) const {
	out << "I_{" << _order << "}";
}

Expression * IdentityMatrix::Clone() const {
	return GetIdentityMatrix(_order);
}

Expression * IdentityMatrix::Differentiate() const {
	throw std::logic_error("differentiating an identity matrix");
}

Expression * IdentityMatrix::Vectorize() const {
	throw std::logic_error("vectorizing an identity matrix");
}

ExpressionType CommutationMatrix::GetExpressionType() const {
	return ExpressionType::kCommutationMatrix;
}

Expression* CommutationMatrix::GetCommutationMatrix(int m, int n) {
	if (m == 1 || n == 1) {
		return IdentityMatrix::GetIdentityMatrix(m * n);
	} else {
		return new CommutationMatrix(
			m * n, m * n,
			false, false, m, n
		);
	}
}

void CommutationMatrix::Print(std::ostream &out) const {
	out << "K_{" << _m << ", " << _n << "}";
}

Expression * CommutationMatrix::Clone() const {
	return GetCommutationMatrix(_m, _n);
}

Expression * CommutationMatrix::Differentiate() const {
	throw std::logic_error("differentiating a commutation matrix");
}

Expression * CommutationMatrix::Vectorize() const {
	throw std::logic_error("vectorizing a commutation matrix");
}

ExpressionType ScalarConstant::GetExpressionType() const {
	return ExpressionType::kScalarConstant;
}

ScalarConstant* ScalarConstant::GetScalarConstant(double value) {
	return new ScalarConstant(1, 1, false, false, value);
}

void ScalarConstant::Print(std::ostream &out) const {
	out << _value;
}

Expression * ScalarConstant::Clone() const {
	return ScalarConstant::GetScalarConstant(_value);
}

Expression * ScalarConstant::Differentiate() const {
	throw std::logic_error("differentiating a scalar constant");
}

Expression * ScalarConstant::Vectorize() const {
	throw std::logic_error("vectorizing a scalar constant");
}

ExpressionType Variable::GetExpressionType() const {
	return ExpressionType::kVariable;
}

void Variable::MarkVariable(unsigned int uuid) {
	_has_variable = (uuid == _uuid);
}

Expression* Variable::GetDerivative() const {
	if (_has_differential) {
		return IdentityMatrix::GetIdentityMatrix(_rows);
	} else {
		throw std::logic_error("getting derivatives on expressions without differential");
	}
}

Variable* Variable::GetVariable(const std::string &name, unsigned int uuid, int rows, int cols) {
	return new Variable(rows, cols, false, false, name, uuid);
}

void Variable::Print(std::ostream &out) const {
	if (_has_differential) {
		out << "d";
	}
	out << _name;
}

Expression * Variable::Clone() const {
	return new Variable(_rows, _cols, _has_variable, _has_differential, _name, _uuid);
}

Expression * Variable::Differentiate() const {
	auto new_var = GetVariable(_name, _uuid, _rows, _cols);
	new_var->_has_differential = true;
	return new_var;
}

Expression * Variable::Vectorize() const {
	return new Variable(_rows * _cols, 1, _has_variable, _has_differential, _name, _uuid);
}

unsigned int UUIDGenerator::_cnt = 0;

unsigned int UUIDGenerator::GenUUID() {
	return _cnt++;
}

}