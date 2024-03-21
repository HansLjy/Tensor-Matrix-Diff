#include "Expressions.hpp"
#include <iostream>
#include <cassert>

namespace TMD {

Negative* Negative::GetNegativeExpression(Expression *child) {
	return new Negative(child->_rows, child->_cols, child->_has_variable, child->_has_differential, child);
}

void Negative::Print(std::ostream &out) const {
	out << "-\\left\\( ";
	_child->Print(out);
	out << "\\right\\)";
}

Expression* Negative::Clone() const {
	return GetNegativeExpression(_child->Clone());
}

Expression* Negative::Differentiate() {
	_child = _child->Differentiate();
	return this;
}

Expression* Negative::Vectorize() {
	_child = _child->Vectorize();
	return this;
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

Expression* Inverse::Differentiate() {
	auto child_copy1 = _child->Clone();
	auto child_copy2 = _child->Clone();
	auto inv_child_copy1 = GetInverseExpression(child_copy1);
	auto inv_child_copy2 = GetInverseExpression(child_copy2);
	auto neg_inv_child_copy1 = Negative::GetNegativeExpression(inv_child_copy1);
	auto mult1 = MatrixMultiplication::GetMatrixMultiplicationExpression(neg_inv_child_copy1, _child->Differentiate());
	auto result = MatrixMultiplication::GetMatrixMultiplicationExpression(mult1, inv_child_copy2);
	return result;
}

Expression* Inverse::Vectorize() {
	throw std::logic_error("vectorizing the inverse");
}

MatrixDeterminant* MatrixDeterminant::GetMatrixDeterminantExpression(Expression *child) {
	return new MatrixDeterminant(
		1, 1, child->_has_variable, child->_has_differential, child
	);
}



MatrixMultiplication* MatrixMultiplication::GetMatrixMultiplicationExpression(
	Expression *lhs, Expression *rhs
) {
	assert(lhs->_cols == rhs->_rows);
	return new MatrixMultiplication(
		lhs->_rows, rhs->_cols,
		lhs->_has_variable || rhs->_has_variable,
		lhs->_has_differential || rhs->_has_differential,
		lhs, rhs
	);
}

}