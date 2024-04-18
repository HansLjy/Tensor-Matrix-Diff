#include "Expressions.hpp"
#include <iostream>
#include <cassert>

#ifdef IMPLEMENT_SLOW_EVALUATION
	#include "unsupported/Eigen/KroneckerProduct"
	#include "MatrixManipulation.hpp"
	#include "FiniteDifference.hpp"
#endif

namespace TMD {

ExpressionPtr GetDerivative(const ExpressionPtr expression, unsigned int variable) {
	auto expr = expression;
	expr = expr->MarkVariable(variable);
	if (!expr->_has_variable) {
		return nullptr;
	}
	expr = expr->Differentiate();
	expr = expr->MarkDifferential();
	expr = expr->Vectorize();
	auto result = Transpose::GetSelfType(
		expr->GetTransposedDerivative()
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

std::map<unsigned int, int> Expression::CountInDegree() const {
	std::map<unsigned int, int> in_degrees;
	std::set<unsigned int> visited;
	RealCountInDegree(visited, in_degrees);
	return in_degrees;
}

std::string Expression::ExportGraph() const {
	std::stringstream out;

	out << "% !TeX TXS-program:compile = txs:///pdflatex/[--shell-escape]" << std::endl;
	out << "\\documentclass{standalone}" << std::endl
	    << "\\usepackage[psfrag]{graphviz}" << std::endl
		<< "\\usepackage{auto-pst-pdf}" << std::endl
		<< "\\usepackage{psfrag}" << std::endl;
	
	out << "\\begin{document}" << std::endl;

	std::map<unsigned int, unsigned int> duplicated_expr_ids;
	int duplicated_expr_cnt = 0;
	
	auto in_degrees = CountInDegree();
	for (const auto [uuid, in_degree] : in_degrees) {
		if (in_degree > 1) {
			duplicated_expr_ids[uuid] = duplicated_expr_cnt++;
		}
	}

	std::vector<bool> duplicated_expr_exported(duplicated_expr_cnt, false);
	std::vector<std::string> labels;

	std::stringstream real_out;
	int tree_cnt = 1;
	int tree_node_cnt = 0;
	RealExportGraph(tree_cnt, 0, tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, real_out);

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

void Expression::RealExportGraph(
	int &tree_cnt,
	int cur_tree_id,
	int &tree_node_cnt,
	const std::map<unsigned int, unsigned int> &duplicated_expr_ids,
	std::vector<bool> &duplicated_expr_exported,
	std::vector<std::string> &labels,
	std::stringstream &out
) const {
	auto itr = duplicated_expr_ids.find(_uuid);
	if (itr != duplicated_expr_ids.end()) {
		const auto new_id = itr->second;
		out << "nd" << cur_tree_id << "_" <<  tree_node_cnt << "[label = EXPR" << new_id << "]" << std::endl;
		tree_node_cnt++;
		if (!duplicated_expr_exported[new_id]) {
			cur_tree_id = tree_cnt++;
			int new_tree_node_cnt = 0;
			GetExportedGraph(tree_cnt, cur_tree_id, new_tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, out);
			out << "nd" << cur_tree_id << "[label = EXPR" << new_id << "]" << std::endl;
			out << "nd" << cur_tree_id << " -> " << "nd" << cur_tree_id << "_" << new_tree_node_cnt - 1 << ";" << std::endl;
			duplicated_expr_exported[new_id] = true;
		}
	} else {
		GetExportedGraph(tree_cnt, cur_tree_id, tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, out);
	}
}

Expression::Expression(
	int priority,
	const ExpressionType& type,
	int rows, int cols,
	bool has_variable,
	bool has_differential):
	_uuid(internal::UUIDGenerator::GenUUID()),
	_priority(priority),
	_type(type),
	_rows(rows), _cols(cols),
	_has_variable(has_variable),
	_has_differential(has_differential) {}

ExpressionPtr Expression::Substitute(
	const std::map<unsigned int, ExpressionPtr> &subs
) {
	std::map<unsigned int, ExpressionPtr> subed_exprs;
	return RealSubstitute(subs, subed_exprs);
}

ExpressionPtr Expression::RealSubstitute(
	const std::map<unsigned int, ExpressionPtr> &subs,
	std::map<unsigned int, ExpressionPtr> &subed_exprs
) {
	auto itr = subed_exprs.find(_uuid);
	if (itr != subed_exprs.end()) {
		return itr->second;
	} else {
		auto new_expr = GetSubedExpression(subs, subed_exprs);
		subed_exprs[_uuid] = new_expr;
		return new_expr;
	}
}

ExpressionPtr Expression::GetTransposedDerivative() const {
	throw std::logic_error("should be copying instead of getting derivative");
}

ExpressionPtr Expression::MarkVariable(unsigned int variable_id) const {
	std::map<unsigned int, ExpressionPtr> marked_exprs;
	return RealMarkVariable(variable_id, marked_exprs);
}

ExpressionPtr Expression::RealMarkVariable(
	unsigned int variable_id,
	std::map<unsigned int, ExpressionPtr> &marked_exprs
) const {
	auto itr = marked_exprs.find(_uuid);
	if (itr != marked_exprs.end()) {
		return itr->second;
	} else {
		auto new_expr = GetMarkedVariableExpression(variable_id, marked_exprs);
		marked_exprs[_uuid] = new_expr;
		return new_expr;
	}
}

ExpressionPtr Expression::Differentiate() const {
	std::map<unsigned int, ExpressionPtr> diffed_exprs;
	return RealDifferentiate(diffed_exprs);
}

ExpressionPtr Expression::RealDifferentiate(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	auto itr = diffed_exprs.find(_uuid);
	if (itr != diffed_exprs.end()) {
		return itr->second;
	} else {
		auto new_expr = GetDiffedExpression(diffed_exprs);
		diffed_exprs[_uuid] = new_expr;
		return new_expr;
	}
}

ExpressionPtr Expression::MarkDifferential() {
	std::map<unsigned int, ExpressionPtr> marked_exprs;
	return RealMarkDifferential(marked_exprs);
}

ExpressionPtr Expression::RealMarkDifferential(std::map<unsigned int, ExpressionPtr> &marked_exprs) {
	auto itr = marked_exprs.find(_uuid);
	if (itr != marked_exprs.end()) {
		return itr->second;
	} else {
		auto new_expr = GetMarkedDifferentialExpression(marked_exprs);
		marked_exprs[_uuid] = new_expr;
		return new_expr;
	}
}

ExpressionPtr Expression::Vectorize() const {
	std::map<unsigned int, ExpressionPtr> veced_exprs;
	return RealVectorize(veced_exprs);
}

ExpressionPtr Expression::RealVectorize(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	auto itr = veced_exprs.find(_uuid);
	if (itr != veced_exprs.end()) {
		return itr->second;
	} else {
		auto new_expr = GetVecedExpression(veced_exprs);
		veced_exprs[_uuid] = new_expr;
		return new_expr;
	}
}

ExpressionPtr SingleOpExpression::GetMarkedVariableExpression(
	unsigned int variable_id,
	std::map<unsigned int, ExpressionPtr> &marked_exprs
) const {
	return GetSingleOpExpression(_child->RealMarkVariable(variable_id, marked_exprs));
}

ExpressionPtr SingleOpExpression::GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr> &marked_exprs) {
	auto new_child = _child->RealMarkDifferential(marked_exprs);
	if (new_child != _child || _has_differential != _child->_has_differential) {
		auto new_expr = GetSingleOpExpression(new_child);
		return new_expr;
	} else {
		return shared_from_this();
	}
}

ExpressionPtr SingleOpExpression::GetSubedExpression(const std::map<unsigned int, ExpressionPtr> &subs, std::map<unsigned int, ExpressionPtr> &subed_exprs) {
	auto new_child = _child->RealSubstitute(subs, subed_exprs);
	if (new_child != _child) {
		return GetSingleOpExpression(new_child);
	} else {
		return shared_from_this();
	}
}

void SingleOpExpression::CollectExpressionIds(std::vector<unsigned int> &ids) const {
	ids.push_back(_uuid);
	_child->CollectExpressionIds(ids);
}

void SingleOpExpression::GetExportedGraph(
	int &tree_cnt,
	int cur_tree_id,
	int& tree_node_cnt,
	const std::map<unsigned int, unsigned int> &duplicated_expr_ids,
	std::vector<bool> &duplicated_expr_exported,
	std::vector<std::string> &labels,
	std::stringstream &out
) const {
	const int label_id = labels.size();
	labels.push_back(_operator_sym);
	_child->RealExportGraph(tree_cnt, cur_tree_id, tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, out);
	const int child_id = tree_node_cnt - 1;
	out << "nd" << cur_tree_id << "_" << tree_node_cnt << "[label = label" << label_id << "]" << std::endl;
	out << "nd" << cur_tree_id << "_" << tree_node_cnt << " -> " << "nd" << cur_tree_id << "_" << child_id << ";" << std::endl;
	tree_node_cnt++;
}

void SingleOpExpression::RealCountInDegree(
	std::set<unsigned int> &visited,
	std::map<unsigned int, int> &in_degrees
) const {
	if (visited.find(_uuid) != visited.end()) {
		return;
	}
	in_degrees[_child->_uuid]++;
	visited.insert(_uuid);
	_child->RealCountInDegree(visited, in_degrees);
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

ExpressionPtr Negate::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return GetSelfType(_child->RealDifferentiate(diffed_exprs));
}

ExpressionPtr Negate::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
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

ExpressionPtr Inverse::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	auto inv_child1 = GetSelfType(_child);
	auto inv_child2 = GetSelfType(_child);
	auto neg_inv_child1 = Negate::GetSelfType(inv_child1);
	auto mult1 = MatrixProduct::GetSelfType(neg_inv_child1, _child->RealDifferentiate(diffed_exprs));
	auto result = MatrixProduct::GetSelfType(mult1, inv_child2);
	return result;
}

ExpressionPtr Inverse::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
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

ExpressionPtr Determinant::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	auto det_A = GetSelfType(_child);
	auto trans_inv_A = Transpose::GetSelfType(Inverse::GetSelfType(_child));
	auto vec_inv_A = Vectorization::GetSelfType(trans_inv_A);
	auto trans_vec_inv_A = Transpose::GetSelfType(vec_inv_A);
	auto vec_diff_A = Vectorization::GetSelfType(_child->RealDifferentiate(diffed_exprs));
	auto mult1 = MatrixProduct::GetSelfType(trans_vec_inv_A, vec_diff_A);
	auto result = ScalarMatrixProduct::GetSelfType(det_A, mult1);
	return result;
}

ExpressionPtr Determinant::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
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

ExpressionPtr Vectorization::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return GetSelfType(_child->RealDifferentiate(diffed_exprs));
}

ExpressionPtr Vectorization::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	return _child->RealVectorize(veced_exprs);
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

ExpressionPtr Transpose::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return GetSelfType(_child->RealDifferentiate(diffed_exprs));
}

ExpressionPtr Transpose::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	auto K = CommutationMatrix::GetSelfType(_child->_rows, _child->_cols);
	auto vec_child = _child->RealVectorize(veced_exprs);
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

ExpressionPtr Diagonalization::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return GetSelfType(_child->RealDifferentiate(diffed_exprs));
}

ExpressionPtr Diagonalization::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	return MatrixProduct::GetSelfType(
		DiagonalizationMatrix::GetSelfType(_child->_rows),
		_child->RealVectorize(veced_exprs)
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

ExpressionPtr Skew::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return GetSelfType(_child->RealDifferentiate(diffed_exprs));
}

ExpressionPtr Skew::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	return MatrixProduct::GetSelfType(
		SkewMatrix::GetSelfType(),
		_child->RealVectorize(veced_exprs)
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

ExpressionPtr Exp::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	return HadamardProduct::GetSelfType(
		GetSelfType(_child),
		_child->RealDifferentiate(diffed_exprs)
	);
}

ExpressionPtr Exp::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	throw std::logic_error("vectorizing an exp expression");
}

ExpressionPtr Exp::GetSelfType(ExpressionPtr child) {
	return std::make_shared<Exp>(
		child->_rows, child->_cols,
		child->_has_variable, child->_has_differential,
		child
	);
}

ExpressionPtr DoubleOpExpression::GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr> &marked_exprs) const {
	return GetDoubleOpExpression(
		_lhs->RealMarkVariable(variable_id, marked_exprs),
		_rhs->RealMarkVariable(variable_id, marked_exprs)
	);
}

ExpressionPtr DoubleOpExpression::GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr> &marked_exprs) {
	auto new_lhs = _lhs->RealMarkDifferential(marked_exprs);
	auto new_rhs = _rhs->RealMarkDifferential(marked_exprs);
	if (new_lhs != _lhs || new_rhs != _rhs || _has_differential != (new_lhs->_has_differential || new_rhs->_has_differential)) {
		auto new_expr = GetDoubleOpExpression(new_lhs, new_rhs);
		return new_expr;
	} else {
		return shared_from_this();
	}
}

ExpressionPtr DoubleOpExpression::GetSubedExpression(
	const std::map<unsigned int, ExpressionPtr> &subs,
	std::map<unsigned int, ExpressionPtr> &subed_exprs
) {
	auto new_lhs = _lhs->RealSubstitute(subs, subed_exprs);
	auto new_rhs = _rhs->RealSubstitute(subs, subed_exprs);
	if (new_lhs != _lhs || new_rhs != _rhs) {
		return GetDoubleOpExpression(new_lhs, new_rhs);
	} else {
		return shared_from_this();
	}
}

void DoubleOpExpression::CollectExpressionIds(std::vector<unsigned int> &ids) const {
	ids.push_back(_uuid);
	_lhs->CollectExpressionIds(ids);
	_rhs->CollectExpressionIds(ids);
}

void DoubleOpExpression::RealCountInDegree(
	std::set<unsigned int> &visited,
	std::map<unsigned int, int> &in_degrees
) const {
	if (visited.find(_uuid) != visited.end()) {
		return;
	}
	visited.insert(_uuid);
	in_degrees[_lhs->_uuid]++;
	in_degrees[_rhs->_uuid]++;
	_lhs->RealCountInDegree(visited, in_degrees);
	_rhs->RealCountInDegree(visited, in_degrees);
}

void DoubleOpExpression::GetExportedGraph(
	int &tree_cnt,
	int cur_tree_id,
	int& tree_node_cnt,
	const std::map<unsigned int, unsigned int> &duplicated_expr_ids,
	std::vector<bool> &duplicated_expr_exported,
	std::vector<std::string> &labels,
	std::stringstream &out
) const {
	const int label_id = labels.size();
	labels.push_back(_operator_sym);
	_lhs->RealExportGraph(tree_cnt, cur_tree_id, tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, out);
	const int lhs_node_id = tree_node_cnt - 1;
	_rhs->RealExportGraph(tree_cnt, cur_tree_id, tree_node_cnt, duplicated_expr_ids, duplicated_expr_exported, labels, out);
	const int rhs_node_id = tree_node_cnt - 1;

	out << "nd" << cur_tree_id << "_" << tree_node_cnt << "[label = label" << label_id << "]" << std::endl;
	out << "nd" << cur_tree_id << "_" << tree_node_cnt << " -> " << "nd" << cur_tree_id << "_" << lhs_node_id << ";" << std::endl;
	out << "nd" << cur_tree_id << "_" << tree_node_cnt << " -> " << "nd" << cur_tree_id << "_" << rhs_node_id << ";" << std::endl;
	tree_node_cnt++;
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

ExpressionPtr MatrixAddition::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs->RealDifferentiate(diffed_exprs));
	} else if (_lhs->_has_variable && !_rhs->_has_variable) {
		return _lhs->RealDifferentiate(diffed_exprs);
	} else if (!_lhs->_has_variable && _rhs->_has_variable) {
		return _rhs->RealDifferentiate(diffed_exprs);
	} else {
		throw std::logic_error("differentiating an expression without variable");
	}
}

ExpressionPtr MatrixAddition::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	if (_lhs->_has_differential && _rhs->_has_differential) {
		return GetSelfType(_lhs->RealVectorize(veced_exprs), _rhs->RealVectorize(veced_exprs));
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
		return GetSelfType(_lhs, _rhs->GetTransposedDerivative());
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

ExpressionPtr MatrixProduct::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs);
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs));
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs),
			GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs))
		);
	}
	throw std::logic_error("differentiating a matrix product with no varaible");
}

ExpressionPtr MatrixProduct::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	// A * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto BT = Transpose::GetSelfType(_rhs);
		auto BTKronI = KroneckerProduct::GetSelfType(
			BT, IdentityMatrix::GetSelfType(_lhs->_rows)
		);
		return MatrixProduct::GetSelfType(
			BTKronI, _lhs->RealVectorize(veced_exprs)
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		auto IKronA = KroneckerProduct::GetSelfType(
			IdentityMatrix::GetSelfType(_rhs->_cols), _lhs
		);
		return MatrixProduct::GetSelfType(
			IKronA, _rhs->RealVectorize(veced_exprs)
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

ExpressionPtr KroneckerProduct::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs);
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs));
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs),
			GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs))
		);
	}
	throw std::logic_error("differentiating a Kronecker product with no varaible");
}

ExpressionPtr KroneckerProduct::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
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
			Vectorization::GetSelfType(_lhs),
			IdentityMatrix::GetSelfType(_rhs->_rows * _rhs->_cols)
		);
		return MatrixProduct::GetSelfType(
			I_kron_K_kron_I,
			MatrixProduct::GetSelfType(
				vec_A_kron_I, _rhs->RealVectorize(veced_exprs)
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
			Vectorization::GetSelfType(_rhs),
			IdentityMatrix::GetSelfType(_lhs->_rows * _lhs->_cols)
		);
		return MatrixProduct::GetSelfType(
			K_kron_K,
			MatrixProduct::GetSelfType(
				I_kron_K_kron_I,
				MatrixProduct::GetSelfType(
					vec_B_kron_I, _lhs->RealVectorize(veced_exprs)
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

ExpressionPtr ScalarMatrixProduct::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	if (_lhs->_has_variable && !_rhs->_has_variable) {
		return GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs);
	}
	if (_rhs->_has_variable && !_lhs->_has_variable) {
		return GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs));
	}
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			GetSelfType(_lhs->RealDifferentiate(diffed_exprs), _rhs),
			GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs))
		);
	}
	throw std::logic_error("differentiating a scalar-matrix product with no varaible");
}

ExpressionPtr ScalarMatrixProduct::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	// a * B
	if (_lhs->_has_differential && _rhs->_has_differential) {
		throw std::logic_error("differential appears on both sides of scalar-matrix product");
	}
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		auto vec_B = Vectorization::GetSelfType(_rhs);
		return MatrixProduct::GetSelfType(
			vec_B, _lhs->RealVectorize(veced_exprs)
		);
	}
	if (!_lhs->_has_differential && _rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			KroneckerProduct::GetSelfType(_lhs, IdentityMatrix::GetSelfType(_rhs->_rows * _rhs->_cols)),
			_rhs->RealVectorize(veced_exprs)
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

ExpressionPtr MatrixScalarPower::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	ExpressionPtr power_minus_1;
	if (_rhs->_type == ExpressionType::kRationalScalarConstant) {
		std::shared_ptr<RationalScalarConstant> rhs_copy = 
			std::dynamic_pointer_cast<RationalScalarConstant>(_rhs->Clone());
		rhs_copy->_value += -1;
		power_minus_1 = rhs_copy;
	} else {
		power_minus_1 = MatrixAddition::GetSelfType(
			_rhs,
			RationalScalarConstant::GetSelfType(-1)
		);
	}
	return HadamardProduct::GetSelfType(
		ScalarMatrixProduct::GetSelfType(
			_rhs,
			GetSelfType(_lhs, power_minus_1)
		),
		_lhs->RealDifferentiate(diffed_exprs)
	);
}

ExpressionPtr MatrixScalarPower::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	throw std::logic_error("vectorizing a matrix scalar power");
}

ExpressionPtr HadamardProduct::Clone() const {
	return GetSelfType(_lhs->Clone(), _rhs->Clone());
}

ExpressionPtr HadamardProduct::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	if (_lhs->_has_variable && _rhs->_has_variable) {
		return MatrixAddition::GetSelfType(
			HadamardProduct::GetSelfType(_rhs, _lhs->RealDifferentiate(diffed_exprs)),
			HadamardProduct::GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs))
		);
	} else if (_lhs->_has_variable && !_rhs->_has_variable) {
		return HadamardProduct::GetSelfType(_rhs, _lhs->RealDifferentiate(diffed_exprs));
	} else if (!_lhs->_has_variable && _rhs->_has_variable) {
		return HadamardProduct::GetSelfType(_lhs, _rhs->RealDifferentiate(diffed_exprs));
	} else {
		throw std::logic_error("differentiating a hadamard product with no variable");
	}
}

ExpressionPtr HadamardProduct::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	if (_lhs->_has_differential && !_rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			Diagonalization::GetSelfType(
				Vectorization::GetSelfType(
					_rhs
				)
			),
			_lhs->RealVectorize(veced_exprs)
		);
	} else if (!_lhs->_has_differential && _rhs->_has_differential) {
		return MatrixProduct::GetSelfType(
			Diagonalization::GetSelfType(
				Vectorization::GetSelfType(
					_lhs
				)
			),
			_rhs->RealVectorize(veced_exprs)
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

ExpressionPtr LeafExpression::GetMarkedVariableExpression(
	unsigned int variable_id,
	std::map<unsigned int, ExpressionPtr> &marked_exprs
) const {
	auto new_expr = Clone();
	new_expr->_has_variable = false;
	return new_expr;
}

ExpressionPtr LeafExpression::GetMarkedDifferentialExpression(
	std::map<unsigned int, ExpressionPtr> &marked_exprs
) {
	return shared_from_this();
}

ExpressionPtr LeafExpression::GetSubedExpression(
	const std::map<unsigned int, ExpressionPtr> &subs,
	std::map<unsigned int, ExpressionPtr> &subed_exprs
) {
	return shared_from_this();
}

void LeafExpression::CollectExpressionIds(std::vector<unsigned int> &ids) const {
	// we do not collect uuids of the leaves so that they never get aliased
}

void LeafExpression::GetExportedGraph(
	int &tree_cnt,
	int cur_tree_id,
	int &tree_node_cnt,
	const std::map<unsigned int, unsigned int> &duplicated_expr_ids,
	std::vector<bool> &duplicated_expr_exported,
	std::vector<std::string> &labels,
	std::stringstream &out
) const {
	const int label_id = labels.size();
	std::stringstream cur_out;
	Print(cur_out);
	labels.push_back(cur_out.str());
	out << "nd" << cur_tree_id << "_" << tree_node_cnt << "[label = label" << label_id << "]" << std::endl;
	tree_node_cnt++;
}

void LeafExpression::RealCountInDegree(
	std::set<unsigned int> &visited,
	std::map<unsigned int, int> &in_degrees
) const {
	in_degrees[_uuid] = 0;
	return;
}

ExpressionPtr LeafExpression::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	throw std::logic_error("differentiating a leaf expression without variable");
}

ExpressionPtr LeafExpression::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
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
	if (element_row >= rows || element_col >= cols) {
		throw std::logic_error("invalid element matrix");
	}
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

ExpressionPtr Variable::GetMarkedVariableExpression(
	unsigned int variable_id,
	std::map<unsigned int, ExpressionPtr> &marked_exprs
) const {
	auto new_expr = Clone();
	new_expr->_has_variable = (variable_id == _variable_id);
	return new_expr;
}

ExpressionPtr Variable::GetSubedExpression(
	const std::map<unsigned int, ExpressionPtr> &subs,
	std::map<unsigned int, ExpressionPtr> &subed_exprs
) {
	auto itr = subs.find(_variable_id);
	if (itr != subs.end()) {
		auto new_expr = itr->second;
		if (_rows != new_expr->_rows || _cols != new_expr->_cols) {
			throw std::logic_error("substitution does not match in size");
		}
		return new_expr;
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

std::shared_ptr<Variable> Variable::GetSelfType(const std::string &name, unsigned int variable_id, int rows, int cols) {
	return std::make_shared<Variable>(rows, cols, false, false, name, variable_id);
}

void Variable::Print(std::ostream &out) const {
	if (_has_differential) {
		out << "d";
	}
	out << _name;
}

ExpressionPtr Variable::Clone() const {
	return std::make_shared<Variable>(_rows, _cols, _has_variable, _has_differential, _name, _variable_id);
}

ExpressionPtr Variable::GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) const {
	auto new_var = GetSelfType(_name, _variable_id, _rows, _cols);
	new_var->_has_differential = true;
	return new_var;
}

ExpressionPtr Variable::GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const {
	return std::make_shared<Variable>(_rows * _cols, 1, _has_variable, _has_differential, _name, _variable_id);
}

std::ostream& operator<<(std::ostream& out, const Expression& expr) {
	expr.Print(out);
	return out;
}

namespace internal {

unsigned int UUIDGenerator::_cnt = 0;

unsigned int UUIDGenerator::GenUUID() {
	return _cnt++;
}

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
	result(_element_row, _element_col) = 1;
	return result;
}

Eigen::MatrixXd RationalScalarConstant::SlowEvaluation(const VariableTable& table) const {
	return (Eigen::MatrixXd(1, 1) << static_cast<double>(_value)).finished();
}

Eigen::MatrixXd Variable::SlowEvaluation(const VariableTable& table) const {
	auto itr = table.find(_variable_id);
	if (itr == table.end()) {
		throw std::logic_error("cannot find variable, variable_id = " + std::to_string(_variable_id));
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
	unsigned int variable_id
) {
	TMD::VariableTable tmp_table = table;
	auto func = [&tmp_table, &expression, variable_id] (const Eigen::MatrixXd X) -> Eigen::MatrixXd {
		tmp_table[variable_id] = X;
		return expression->SlowEvaluation(tmp_table);
	};
	return Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func, table.at(variable_id), 1e-4);
}

#endif

}