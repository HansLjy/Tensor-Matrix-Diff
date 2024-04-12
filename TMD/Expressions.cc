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

ExpressionPtr GetDotProduct(const ExpressionPtr lhs, const ExpressionPtr rhs) {
	if (lhs->_rows != rhs->_rows || lhs->_cols != 1 || rhs->_cols != 1) {
		throw std::logic_error("dot producting two matrices with invalid shape");
	}
	return MatrixProduct::GetSelfType(
		Transpose::GetSelfType(lhs),
		rhs
	);
}

ExpressionPtr GetVector2Norm(const ExpressionPtr x) {
	if (x->_cols != 1) {
		throw std::logic_error("getting vector norm of a non-vector");
	}
	return MatrixScalarPower::GetSelfType(
		GetDotProduct(x, x),
		RationalScalarConstant::GetSelfType({1, 2})
	);
}

ExpressionPtr GetMultipleProduct(const std::vector<ExpressionPtr> &prods) {
	if (prods.size() < 2) {
		throw std::logic_error("not enough operands");
	}
	auto result = MatrixProduct::GetSelfType(prods[0], prods[1]);
	for (unsigned int i = 2; i < prods.size(); i++) {
		result = MatrixProduct::GetSelfType(result, prods[i]);
	}
	return result;
}

ExpressionPtr GetMultipleAddition(const std::vector<ExpressionPtr> &adds) {
	if (adds.size() < 2) {
		throw std::logic_error("not enough operands");
	}
	auto result = MatrixAddition::GetSelfType(adds[0], adds[1]);
	for (unsigned int i = 2; i < adds.size(); i++) {
		result = MatrixAddition::GetSelfType(result, adds[i]);
	}
	return result;
}

std::string Expression::ExportGraph() const {
	std::stringstream out;

	out << "% !TeX TXS-program:compile = txs:///pdflatex/[--shell-escape]" << std::endl;
	out << "\\documentclass{standalone}" << std::endl
	    << "\\usepackage[psfrag]{graphviz}" << std::endl
		<< "\\usepackage{auto-pst-pdf}" << std::endl
		<< "\\usepackage{psfrag}" << std::endl;
	
	out << "\\begin{document}" << std::endl;

	std::vector<std::string> labels;
	std::stringstream real_out;
	RealExportGraph(labels, real_out);

	for (unsigned int i = 0; i < labels.size(); i++) {
		out << "\\psfrag{" << "label" << i << "}[cc][cc]{$"
			<< labels[i] << "$}" << std::endl;
	}

	out << "\\digraph{abc}{" << std::endl;
	out << real_out.str();
	out << "}" << std::endl;

	out << "\\end{document}" << std::endl;

	return out.str();
}

ExpressionPtr Expression::Substitute(
	const std::map<unsigned int, const ExpressionPtr> &subs
) const {
	auto copy = Clone();
	// TODO: make this more efficient
	for (const auto& [uuid, expr] : subs){
		copy = copy->RealSubstitute(uuid, expr);
	}
	return copy;
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

ExpressionPtr SingleOpExpression::RealSubstitute(unsigned int uuid, const ExpressionPtr expr) {
	_child = _child->RealSubstitute(uuid, expr);
	return shared_from_this();
}

void SingleOpExpression::RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const {
	unsigned int cur_node_id = labels.size();
	out << "nd" << cur_node_id << "[label = label" << cur_node_id << "]" << std::endl;
	labels.push_back(_operator_sym);
	unsigned int child_node_id = labels.size();
	_child->RealExportGraph(labels, out);
	out << "nd" << cur_node_id << " -> " << "nd" << child_node_id << ";" << std::endl;
}

ExpressionPtr Negate::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Negate>(
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
		1, 1,
		child->_has_variable, child->_has_differential,
		child
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
		child->_rows * child->_cols, 1,
		child->_has_variable, child->_has_differential,
		child
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
		child->_cols, child->_rows,
		child->_has_variable, child->_has_differential,
		child
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

void Diagonalization::Print(std::ostream &out) const {
	out << "\\text{diag} \\left(";
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
	if (child->_cols != 1) {
		throw std::logic_error("diagonalizing an non-vector expression");
	}
	return std::make_shared<Diagonalization>(
		child->_rows, child->_rows,
		child->_has_variable, child->_has_differential,
		child
	);
}

ExpressionPtr Skew::GetSelfType(ExpressionPtr child) {
	if (child->_cols != 1 || child->_rows != 3) {
		throw std::logic_error("matrix / vector not skew-able");
	}
	return std::make_shared<Skew>(
		3, 3,
		child->_has_variable, child->_has_differential,
		child
	);
}

void Skew::Print(std::ostream &out) const {
	out << "\\left[";
	_child->Print(out);
	out << "\\right]";
}

ExpressionPtr Skew::Clone() const {
	return GetSelfType(_child->Clone());
}

ExpressionPtr Skew::Differentiate() const {
	return GetSelfType(_child->Differentiate());
}

ExpressionPtr Skew::Vectorize() const {
	return MatrixProduct::GetSelfType(
		SkewMatrix::GetSelfType(), // TODO:
		_child->Vectorize()
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
	return std::make_shared<Exp>(
		child->_rows, child->_cols,
		child->_has_variable, child->_has_differential,
		child
	);
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

ExpressionPtr DoubleOpExpression::RealSubstitute(unsigned int uuid, const ExpressionPtr expr) {
	_lhs = _lhs->RealSubstitute(uuid, expr);
	_rhs = _rhs->RealSubstitute(uuid, expr);
	return shared_from_this();
}

void DoubleOpExpression::RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const {
	unsigned int cur_node_id = labels.size();
	out << "nd" << cur_node_id << "[label = label" << cur_node_id << "]" << std::endl;
	labels.push_back(_operator_sym);
	unsigned int lhs_node_id = labels.size();
	_lhs->RealExportGraph(labels, out);
	unsigned int rhs_node_id = labels.size();
	_rhs->RealExportGraph(labels, out);
	out << "nd" << cur_node_id << " -> " << "nd" << lhs_node_id << ";" << std::endl;
	out << "nd" << cur_node_id << " -> " << "nd" << rhs_node_id << ";" << std::endl;
}

void DoubleOpExpression::Print(std::ostream &out) const {
	if (_lhs->_priority >= _priority) {
		_lhs->Print(out);
	} else {
		out << "\\left(";
		_lhs->Print(out);
		out << "\\right)";
	}
	if (_operator_sym.length() > 0) {
		out << " " << _operator_sym << " ";
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
		" ",
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

ExpressionPtr MatrixScalarPower::GetSelfType(
	ExpressionPtr matrix, ExpressionPtr power
) {
	if (power->_rows != 1 || power->_cols != 1) {
		throw std::logic_error("power is not a scalar");
	}
	return std::make_shared<MatrixScalarPower>(
		kMatrixScalarPowerPriority,
		ExpressionType::kMatrixScalarPower,
		"\\text{pow}",
		matrix->_rows, matrix->_cols,
		matrix->_has_variable, matrix->_has_differential,
		matrix, power
	);
}

void MatrixScalarPower::Print(std::ostream &out) const {
	out << "\\left(";
	_lhs->Print(out);
	out << "\\right)^{";
	_rhs->Print(out);
	out << "}";
}

ExpressionPtr MatrixScalarPower::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr MatrixScalarPower::Differentiate() const {
	ExpressionPtr power_minus_1;
	if (_rhs->_type == ExpressionType::kRationalScalarConstant) {
		std::shared_ptr<RationalScalarConstant> rhs_copy = 
			std::dynamic_pointer_cast<RationalScalarConstant>(_rhs->Clone());
		rhs_copy->_value += -1;
		power_minus_1 = rhs_copy;
	} else {
		power_minus_1 = MatrixAddition::GetSelfType(
			_rhs->Clone(),
			RationalScalarConstant::GetSelfType(-1)
		);
	}
	return HadamardProduct::GetSelfType(
		ScalarMatrixProduct::GetSelfType(
			_rhs->Clone(),
			GetSelfType(_lhs, power_minus_1)
		),
		_lhs->Differentiate()
	);
}

ExpressionPtr MatrixScalarPower::Vectorize() const {
	throw std::logic_error("vectorizing a matrix scalar power");
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

ExpressionPtr LeafExpression::RealSubstitute(unsigned int uuid, const ExpressionPtr expr) {
	return shared_from_this();
}

void LeafExpression::RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const {
	unsigned int cur_node_id = labels.size();
	std::stringstream cur_out;
	Print(cur_out);
	labels.push_back(cur_out.str());
	out << "nd" << cur_node_id << "[label = label" << cur_node_id << "]" << std::endl;
}

ExpressionPtr LeafExpression::Differentiate() const {
	throw std::logic_error("differentiating a leaf expression without variable");
}

ExpressionPtr LeafExpression::Vectorize() const {
	throw std::logic_error("vectorizing a leaf expression without differential");
}

ExpressionPtr IdentityMatrix::GetSelfType(int order) {
	return std::make_shared<IdentityMatrix>(order, order, false, false, order);
}

void IdentityMatrix::Print(std::ostream &out) const {
	out << "I_{" << _order << "}";
}

ExpressionPtr IdentityMatrix::Clone() const {
	return GetSelfType(_order);
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

ExpressionPtr DiagonalizationMatrix::GetSelfType(int order) {
	return std::make_shared<DiagonalizationMatrix>(order * order, order, false, false, order);
}

void DiagonalizationMatrix::Print(std::ostream &out) const {
	out << "D_{" << _order << "}";
}

ExpressionPtr DiagonalizationMatrix::Clone() const {
	return GetSelfType(_order);
}

ExpressionPtr SkewMatrix::GetSelfType() {
	return std::make_shared<SkewMatrix>();
}

void SkewMatrix::Print(std::ostream &out) const {
	out << "H";
}

ExpressionPtr SkewMatrix::Clone() const {
	return SkewMatrix::GetSelfType();
}

ExpressionPtr ElementMatrix::GetSelfType(int element_row, int element_col, int rows, int cols) {
	return std::make_shared<ElementMatrix>(element_row, element_col, rows, cols);
}

void ElementMatrix::Print(std::ostream &out) const {
	out << "E_{" << _element_row << ", " << _element_col << "}^{" << _rows << ", " << _cols << "}";
}

ExpressionPtr ElementMatrix::Clone() const {
	return GetSelfType(_element_row, _element_col, _rows, _cols);
}

int gcd(int a, int b) {
	return b > 0 ? gcd(b, a % b) : a;
}

RationalScalarConstant::RationalNumber::RationalNumber(int val): _p(val), _q(1){}
RationalScalarConstant::RationalNumber::RationalNumber(int p, int q): _p(p), _q(q){}

RationalScalarConstant::RationalNumber&
RationalScalarConstant::RationalNumber::operator+=(const RationalNumber &rhs) {
	int new_p = _p * rhs._q + _q * rhs._p;
	int new_q = _q * rhs._q;

	int gcd_pq = gcd(std::abs(new_p), new_q);
	_p = new_p / gcd_pq;
	_q = new_q / gcd_pq;
	return *this;
}

RationalScalarConstant::RationalNumber::operator double() const {
	return double(_p) / double(_q);
}

std::ostream& operator<<(std::ostream& out, const RationalScalarConstant::RationalNumber& r) {
	if (r._q == 1) {
		out << r._p;
	} else {
		if (r._p == 0) {
			out << "0";
		} else if (r._p > 0) {
			out << "\\frac{" << r._p << "}{" << r._q << "}";
		} else {
			out << "-\\frac{" << -r._p << "}{" << r._q << "}";
		}
	}
	return out;
}

ExpressionPtr RationalScalarConstant::GetSelfType(const RationalNumber& value) {
	return std::make_shared<RationalScalarConstant>(1, 1, false, false, value);
}

void RationalScalarConstant::Print(std::ostream &out) const {
	out << _value;
}

ExpressionPtr RationalScalarConstant::Clone() const {
	return RationalScalarConstant::GetSelfType(_value);
}

void Variable::MarkVariable(unsigned int uuid) {
	_has_variable = (uuid == _uuid);
}

ExpressionPtr Variable::RealSubstitute(unsigned int uuid, const ExpressionPtr expr) {
	if (_uuid == uuid) {
		if (_rows != expr->_rows || _cols != expr->_cols) {
			throw std::logic_error("substitution does not match in size");
		}
		return expr->Clone();
	} else {
		return shared_from_this();
	}
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

Eigen::MatrixXd Diagonalization::SlowEvaluation(const VariableTable &table) const {
	return _child->SlowEvaluation(table).asDiagonal();
}

Eigen::MatrixXd Skew::SlowEvaluation(const VariableTable &table) const {
	return Numerics::GetHatMatrix(_child->SlowEvaluation(table));
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

Eigen::MatrixXd MatrixScalarPower::SlowEvaluation(const VariableTable &table) const {
	return _lhs->SlowEvaluation(table).array().pow(_rhs->SlowEvaluation(table)(0, 0));
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

Eigen::MatrixXd SkewMatrix::SlowEvaluation(const VariableTable &table) const {
	return Numerics::GetVecHatMatrix();
}

Eigen::MatrixXd ElementMatrix::SlowEvaluation(const VariableTable &table) const {
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(_rows, _cols);
	result(_rows, _cols) = 1;
	return result;
}

Eigen::MatrixXd RationalScalarConstant::SlowEvaluation(const VariableTable& table) const {
	return (Eigen::MatrixXd(1, 1) << static_cast<double>(_value)).finished();
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