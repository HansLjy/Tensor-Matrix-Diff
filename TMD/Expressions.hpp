#pragma once

#include <memory>
#include <string>
#include <map>
#include "TMDConfig.hpp"
#ifdef IMPLEMENT_SLOW_EVALUATION
	#include "Eigen/Eigen"
#endif

namespace TMD {


#ifdef IMPLEMENT_SLOW_EVALUATION
	typedef std::map<unsigned int, Eigen::MatrixXd> VariableTable;
#endif

class Expression;
typedef std::shared_ptr<Expression> ExpressionPtr;

// return nullptr if the expression does not contain the variable
ExpressionPtr GetDerivative(const ExpressionPtr expression, unsigned int variable);

enum class ExpressionType {
	kNegateOp,
	kInverseOp,
	kDeterminantOp,
	kVectorizationOp,
	kTransposeOp,
	kDiagonalizeOp,
	kSkewOp,
	kExpOp,
	kMatrixAdditionOp,
	kMatrixProductOp,
	kKroneckerProductOp,
	kMatrixScalarPower,
	kScalarMatrixProductOp,
	kHadamardProductOp,
	kIdentityMatrix,
	kCommutationMatrix,
	kDiagonalizationMatrix,
	kSkewMatrix,
	kElementMatrix,
	kRationalScalarConstant,
	kVariable
};

/**
 * Leaf > single op > kron > prod = scalar prod > +
 *  4       3          2          1               0
 */

const int kNegatePriority               = 3;
const int kInversePriority              = 3;
const int kDeterminantPriority          = 3;
const int kVectorizationPriority        = 3;
const int kTransposePriority            = 3;
const int kDiagonalizePriority          = 3;
const int kSkewPriority                 = 3;
const int kExpPriority                  = 3;
const int kMatrixAdditionPriority       = 0;
const int kMatrixProductPriority        = 1;
const int kKroneckerProductPriority     = 2;
const int kMatrixScalarPowerPriority    = 3; // TODO:
const int kScalarMatrixProductPriority  = 1;
const int kHadamardProductPriority      = 2;
const int kLeafPriority                 = 4;

ExpressionPtr GetDotProduct(const ExpressionPtr lhs, const ExpressionPtr rhs);
ExpressionPtr GetVector2Norm(const ExpressionPtr x);
ExpressionPtr GetMultipleProduct(const std::vector<ExpressionPtr>& prods);
ExpressionPtr GetMultipleAddition(const std::vector<ExpressionPtr>& adds);

class Expression : public std::enable_shared_from_this<Expression> {
public:
	virtual void MarkVariable(unsigned int uuid) = 0;
	virtual void MarkDifferential() = 0;

	ExpressionPtr Substitute(const std::map<unsigned int, const ExpressionPtr>& subs) const;
	virtual ExpressionPtr RealSubstitute(unsigned int uuid, const ExpressionPtr expr) = 0;

	virtual ExpressionPtr GetTransposedDerivative() const;

	virtual void Print(std::ostream& out) const = 0;
	std::string ExportGraph() const;
	virtual ExpressionPtr Clone() const = 0;

	// return a **new** expression that equals
	// the differential of the current one
	virtual ExpressionPtr Differentiate() const = 0;
	virtual ExpressionPtr Vectorize() const = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	virtual Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const = 0;
#endif

	virtual ~Expression() = default;

	const int _priority;
	const ExpressionType _type;

	int _rows, _cols;           // for scalar expression, simply set these to (1, 1)
	bool _has_variable;
	bool _has_differential;

	Expression(
		int priority,
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential):
		_priority(priority),
		_type(type),
		_rows(rows), _cols(cols),
		_has_variable(has_variable),
		_has_differential(has_differential) {}
	
	virtual void RealExportGraph(std::vector<std::string>& labels, std::stringstream& out) const = 0;
};

class SingleOpExpression : public Expression {
public:
	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;
	ExpressionPtr RealSubstitute(unsigned int uuid, const ExpressionPtr expr) override;

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override = 0;
	ExpressionPtr Vectorize() const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	std::string _operator_sym;
	ExpressionPtr _child;

	SingleOpExpression(
		int priority,
		const ExpressionType& type,
		const std::string& operator_sym,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		Expression(priority, type, rows, cols, has_variable, has_differential),
		_operator_sym(operator_sym),
		_child(child) {}
	
	void RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const override;
};

class Negate : public SingleOpExpression {
public:
	Negate(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kNegatePriority, ExpressionType::kNegateOp, "-", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Inverse : public SingleOpExpression {
public:
	Inverse(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kInversePriority, ExpressionType::kInverseOp, "\\text{inv}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Determinant : public SingleOpExpression {
public:
	Determinant(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kDeterminantPriority, ExpressionType::kDeterminantOp, "\\det", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Vectorization : public SingleOpExpression {
public:
	Vectorization(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kVectorizationPriority, ExpressionType::kDeterminantOp, "\\text{vec}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Transpose : public SingleOpExpression {
public:
	Transpose(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kTransposePriority, ExpressionType::kTransposeOp, "\\top", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Diagonalization : public SingleOpExpression {
public:
	Diagonalization(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kDiagonalizePriority, ExpressionType::kDiagonalizeOp, "\\text{diag}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Skew : public SingleOpExpression {
public:
	Skew(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(
			kSkewPriority, ExpressionType::kSkewOp,
			"\\left[\\right]",
			rows, cols,
			has_variable, has_differential,
			child) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Exp : public SingleOpExpression {
public:
	Exp(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(kExpPriority, ExpressionType::kExpOp, "\\exp", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream& out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);	
};

class DoubleOpExpression : public Expression {
public:
	DoubleOpExpression(
		int priority,
		const ExpressionType& type,
		const std::string& operator_sym,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr lhs,
		ExpressionPtr rhs):
		Expression(priority, type, rows, cols, has_variable, has_differential),
		_lhs(lhs), _rhs(rhs), _operator_sym(operator_sym) {}

	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;
	ExpressionPtr RealSubstitute(unsigned int uuid, const ExpressionPtr expr) override;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override = 0;
	ExpressionPtr Vectorize() const override = 0;

	void RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	ExpressionPtr _lhs, _rhs;
	const std::string _operator_sym;
};

class MatrixAddition : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionPtr GetTransposedDerivative() const override;

	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class MatrixProduct : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionPtr GetTransposedDerivative() const override;

	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class KroneckerProduct : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class ScalarMatrixProduct : public DoubleOpExpression {
public:
	// lhs will be the scalar
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr scalar, ExpressionPtr matrix);
};

// WARNING: for now, the power has to be constant
class MatrixScalarPower : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr matrix, ExpressionPtr power);
};

class HadamardProduct : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;
	
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);

};

class LeafExpression : public Expression {
public:
	using Expression::Expression;

	LeafExpression(
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential) :
		Expression(kLeafPriority, type, rows, cols, has_variable, has_differential) {}

	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;
	ExpressionPtr RealSubstitute(unsigned int uuid, const ExpressionPtr expr) override;

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	void RealExportGraph(std::vector<std::string> &labels, std::stringstream &out) const override;
};

class IdentityMatrix : public LeafExpression {
public:
	IdentityMatrix(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		int order):
		LeafExpression(ExpressionType::kIdentityMatrix, rows, cols, has_variable, has_differential),
		_order(order) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(int order);

	int _order;
};

class CommutationMatrix : public LeafExpression {
public:
	CommutationMatrix(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		int m, int n):
		LeafExpression(ExpressionType::kCommutationMatrix, rows, cols, has_variable, has_differential),
		_m(m), _n(n) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(int m, int n);

	int _m, _n;
};

class DiagonalizationMatrix : public LeafExpression {
public:
	DiagonalizationMatrix(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		int order):
		LeafExpression(ExpressionType::kDiagonalizationMatrix, rows, cols, has_variable, has_differential),
		_order(order) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(int order);

	int _order;
};

class SkewMatrix : public LeafExpression {
public:
	SkewMatrix():
		LeafExpression(ExpressionType::kSkewMatrix, 9, 3, false, false) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType();
};

class ElementMatrix : public LeafExpression {
public:
	ElementMatrix(
		int element_row, int element_col,
		int rows, int cols):
		LeafExpression(ExpressionType::kElementMatrix, rows, cols, false, false) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

	int _element_row, _element_col;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(int element_row, int element_col, int rows, int cols);
};

class RationalScalarConstant : public LeafExpression {
public:
	struct RationalNumber {
		int _p, _q;

		RationalNumber(int val);
		RationalNumber(int p, int q);
		
		RationalNumber& operator+=(const RationalNumber& rhs);

		explicit operator double() const;
	};

	RationalScalarConstant(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		const RationalNumber& val):
		LeafExpression(ExpressionType::kRationalScalarConstant, rows, cols, has_variable, has_differential),
		_value(val) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(const RationalNumber& number);

	RationalNumber _value;
};

class Variable : public LeafExpression {
public:
	Variable(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		const std::string& name,
		unsigned int uuid):
		LeafExpression(ExpressionType::kVariable, rows, cols, has_variable, has_differential),
		_name(name), _uuid(uuid) {}

	void MarkVariable(unsigned int uuid) override;
	ExpressionPtr RealSubstitute(unsigned int uuid, const ExpressionPtr expr) override;

	ExpressionPtr GetTransposedDerivative() const override;
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static std::shared_ptr<Variable> GetSelfType(const std::string& name, unsigned int uuid, int rows, int cols);

	std::string _name;
	unsigned int _uuid;
};

std::ostream& operator<<(std::ostream& out, const Expression& expr);

class UUIDGenerator {
public:
	static unsigned int _cnt;
	static unsigned int GenUUID();
};

#ifdef IMPLEMENT_SLOW_EVALUATION

Eigen::MatrixXd GetExpressionNumericDerivative(
	const TMD::ExpressionPtr expression,
	const TMD::VariableTable& table,
	unsigned int uuid
);

#endif

}