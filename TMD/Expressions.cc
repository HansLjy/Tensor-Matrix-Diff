#include "Expressions.hpp"
#include <iostream>
#include <cassert>

#ifdef IMPLEMENT_SLOW_EVALUATION
	#include "unsupported/Eigen/KroneckerProduct"
	#include "MatrixManipulation.hpp"
#endif

const double eps = 1e-10;

namespace TMD {

Expression* GetDerivative(const Expression* expression, unsigned int variable) {
	auto org_expr = expression->Clone();
	// org_expr->Print(std::cerr);
	// std::cerr << std::endl;
	org_expr->MarkVariable(variable);
	if (!org_expr->_has_variable) {
		return nullptr;
	}

	auto diffed_expr = org_expr->Differentiate();
	diffed_expr->MarkDifferential();
	// diffed_expr->Print(std::cerr);
	// std::cerr << std::endl;
	auto veced_expr = diffed_expr->Vectorize();
	// veced_expr->Print(std::cerr);
	// std::cerr << std::endl;
	auto result = Transpose::GetSelfType(
		veced_expr->GetTransposedDerivative()
	);
	// result->Print(std::cerr);
	// std::cerr << std::endl;

	delete org_expr;
	delete diffed_expr;
	delete veced_expr;
	return result;
}

Expression* Expression::GetTransposedDerivative() const {
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

Negate* Negate::GetSelfType(Expression *child) {
	return new Negate(child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

void Negate::Print(std::ostream &out) const {
	out << "-\\left( ";
	_child->Print(out);
	out << "\\right)";
}

Expression* Negate::Clone() const {
	return GetSelfType(_child->Clone());
}

Expression* Negate::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

Expression* Negate::Vectorize() const {
	return MatrixProduct::GetSelfType(
		Negate::GetSelfType(	
			IdentityMatrix::GetIdentityMatrix(_child->_rows * _child->_cols)
		),
		_child->Vectorize()
	);
	return GetSelfType(_child->Vectorize());
}

ExpressionType Inverse::GetExpressionType() const {
	return ExpressionType::kInverseOp;
}

void Inverse::Print(std::ostream &out) const {
	out << "\\left( ";
	_child->Print(out);
	out << " \\right)^{-1}";
}

Inverse* Inverse::GetSelfType(Expression *child) {
	assert(child->_rows == child->_cols);
	return new Inverse(child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

Expression* Inverse::Clone() const {
	return GetSelfType(_child->Clone());
}

Expression* Inverse::Differentiate() const {
	auto inv_child1 = GetSelfType(_child->Clone());
	auto inv_child2 = GetSelfType(_child->Clone());
	auto neg_inv_child1 = Negate::GetSelfType(inv_child1);
	auto mult1 = MatrixProduct::GetSelfType(neg_inv_child1, _child->Differentiate());
	auto result = MatrixProduct::GetSelfType(mult1, inv_child2);
	return result;
}

Expression* Inverse::Vectorize() const {
	throw std::logic_error("vectorizing the inverse");
}

ExpressionType Determinant::GetExpressionType() const {
	return ExpressionType::kDeterminantOp;
}

Determinant* Determinant::GetSelfType(Expression *child) {
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
	return GetSelfType(_child->Clone());
}

Expression * Determinant::Differentiate() const {
	auto det_A = GetSelfType(_child->Clone());
	auto trans_inv_A = Transpose::GetSelfType(Inverse::GetSelfType(_child->Clone()));
	auto vec_inv_A = Vectorization::GetSelfType(trans_inv_A);
	auto trans_vec_inv_A = Transpose::GetSelfType(vec_inv_A);
	auto vec_diff_A = Vectorization::GetSelfType(_child->Differentiate());
	auto mult1 = MatrixProduct::GetSelfType(trans_vec_inv_A, vec_diff_A);
	auto result = ScalarMatrixProduct::GetSelfType(det_A, mult1);
	return result;
}

Expression * Determinant::Vectorize() const {
	throw std::logic_error("vecorizing the determinant");
}

ExpressionType Vectorization::GetExpressionType() const {
	return ExpressionType::kVectorizationOp;
}

Vectorization* Vectorization::GetSelfType(Expression *child) {
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
	return GetSelfType(_child->Clone());
}

Expression* Vectorization::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

Expression* Vectorization::Vectorize() const {
	return _child->Vectorize();
}

ExpressionType Transpose::GetExpressionType() const {
	return ExpressionType::kTransposeOp;
}

Transpose* Transpose::GetSelfType(Expression *child) {
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
	return GetSelfType(_child->Clone());
}

Expression * Transpose::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

Expression * Transpose::Vectorize() const {
	auto K = CommutationMatrix::GetCommutationMatrix(_child->_rows, _child->_cols);
	auto vec_child = _child->Vectorize();
	return MatrixProduct::GetSelfType(K, vec_child);
}

ExpressionType ScalarPower::GetExpressionType() const {
	return ExpressionType::kScalarPowerOp;
}

ScalarPower* ScalarPower::GetSelfType(Expression *child, double power) {
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
	return GetSelfType(_child->Clone(), _power);
}

Expression * ScalarPower::Differentiate() const {
	auto power = ScalarConstant::GetScalarConstant(_power);
	Expression* mult;
	if (std::abs(_power - 1) < eps) {
		mult  = _child->Differentiate();
	} else {
		auto child_power_m_1 = GetSelfType(_child->Clone(), _power - 1);
		mult = ScalarMatrixProduct::GetSelfType(child_power_m_1, _child->Differentiate());
	}
	return ScalarMatrixProduct::GetSelfType(power, mult);
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

Expression* MatrixAddition::GetTransposedDerivative() const {
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

MatrixAddition* MatrixAddition::GetSelfType(Expression *lhs, Expression *rhs) {
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
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

Expression* MatrixAddition::Differentiate() const {
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

Expression* MatrixAddition::Vectorize() const {
	if (_lhs->_has_differential && _rhs->_has_differential) {
		return GetSelfType(_lhs->Vectorize(), _rhs->Vectorize());
	} else {
		throw std::logic_error("differential appears only on one side of addition");
	}
}

ExpressionType MatrixProduct::GetExpressionType() const {
	return ExpressionType::kMatrixProductOp;
}

Expression* MatrixProduct::GetTransposedDerivative() const {
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

MatrixProduct* MatrixProduct::GetSelfType(
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
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

Expression * MatrixProduct::Differentiate() const {
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

Expression * MatrixProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto BT = Transpose::GetSelfType(_rhs->Clone());
		auto BTKronI = KroneckerProduct::GetSelfType(
			BT, IdentityMatrix::GetIdentityMatrix(_lhs->_rows)
		);
		return MatrixProduct::GetSelfType(
			BTKronI, _lhs->Vectorize()
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto IKronA = KroneckerProduct::GetSelfType(
			IdentityMatrix::GetIdentityMatrix(_rhs->_cols), _lhs->Clone()
		);
		return MatrixProduct::GetSelfType(
			IKronA, _rhs->Vectorize()
		);
	}
	throw std::logic_error("vectorizing a matrix product without differentials");
}

ExpressionType KroneckerProduct::GetExpressionType() const {
	return ExpressionType::kKroneckerProductOp;
}

KroneckerProduct* KroneckerProduct::GetSelfType(Expression *lhs, Expression *rhs) {
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
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

Expression * KroneckerProduct::Differentiate() const {
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

Expression * KroneckerProduct::Vectorize() const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto I_kron_K_kron_I = GetSelfType(
			IdentityMatrix::GetIdentityMatrix(_lhs->_cols),
			GetSelfType(
				CommutationMatrix::GetCommutationMatrix(_rhs->_cols, _lhs->_rows),
				IdentityMatrix::GetIdentityMatrix(_rhs->_rows)
			)
		);
		auto vec_A_kron_I = GetSelfType(
			Vectorization::GetSelfType(_lhs->Clone()),
			IdentityMatrix::GetIdentityMatrix(_rhs->_rows * _rhs->_cols)
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
			CommutationMatrix::GetCommutationMatrix(_lhs->_cols, _rhs->_cols),
			CommutationMatrix::GetCommutationMatrix(_lhs->_rows, _rhs->_rows)
		);
		auto I_kron_K_kron_I = GetSelfType(
			IdentityMatrix::GetIdentityMatrix(_rhs->_cols),
			GetSelfType(
				CommutationMatrix::GetCommutationMatrix(_lhs->_cols, _rhs->_rows),
				IdentityMatrix::GetIdentityMatrix(_lhs->_rows)
			)
		);
		auto vec_B_kron_I = GetSelfType(
			Vectorization::GetSelfType(_rhs->Clone()),
			IdentityMatrix::GetIdentityMatrix(_lhs->_rows * _lhs->_cols)
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

ExpressionType ScalarMatrixProduct::GetExpressionType() const {
	return ExpressionType::kScalarMatrixProductOp;
}

ScalarMatrixProduct* ScalarMatrixProduct::GetSelfType(
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
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

Expression * ScalarMatrixProduct::Differentiate() const {
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

Expression * ScalarMatrixProduct::Vectorize() const {
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
			KroneckerProduct::GetSelfType(_lhs->Clone(), IdentityMatrix::GetIdentityMatrix(_rhs->_rows * _rhs->_cols)),
			_rhs->Vectorize()
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

Expression* Variable::GetTransposedDerivative() const {
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

Eigen::MatrixXd IdentityMatrix::SlowEvaluation(const VariableTable& table) const {
	return Eigen::MatrixXd::Identity(_order, _order);
}

Eigen::MatrixXd CommutationMatrix::SlowEvaluation(const VariableTable& table) const {
	return Numerics::GetPermutationMatrix(_m, _n);
}

Eigen::MatrixXd ScalarConstant::SlowEvaluation(const VariableTable& table) const {
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

#endif

}