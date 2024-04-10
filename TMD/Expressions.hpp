#pragma once

#include <memory>
#include <string>
#include "TMDConfig.hpp"
#ifdef IMPLEMENT_SLOW_EVALUATION
	#include <map>
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
	kScalarPowerOp,
	kDiagonalizeOp,
	kExpOp,
	kMatrixAdditionOp,
	kMatrixProductOp,
	kKroneckerProductOp,
	kScalarMatrixProductOp,
	kHadamardProductOp,
	kIdentityMatrix,
	kCommutationMatrix,
	kDiagonalizationMatrix,
	kScalarConstant,
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
const int kScalarPowerPriority          = 3;
const int kDiagonalizePriority          = 3;
const int kExpPriority                  = 3;
const int kMatrixAdditionPriority       = 0;
const int kMatrixProductPriority        = 1;
const int kKroneckerProductPriority     = 2;
const int kScalarMatrixProductPriority  = 1;
const int kHadamardProductPriority      = 2;
const int kLeafPriority                 = 4;

class Expression {
public:
	virtual void MarkVariable(unsigned int uuid) = 0;
	virtual void MarkDifferential() = 0;

	virtual ExpressionPtr GetTransposedDerivative() const;

	virtual void Print(std::ostream& out) const = 0;
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
};

class SingleOpExpression : public Expression {
public:
	
	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override = 0;
	ExpressionPtr Vectorize() const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	ExpressionPtr _child;

	SingleOpExpression(
		int priority,
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		Expression(priority, type, rows, cols, has_variable, has_differential),
		_child(child) {}
};

class Negate : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

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
	using SingleOpExpression::SingleOpExpression;

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
	using SingleOpExpression::SingleOpExpression;

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
	using SingleOpExpression::SingleOpExpression;

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
	using SingleOpExpression::SingleOpExpression;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class ScalarPower : public SingleOpExpression {
public:
	ScalarPower(
		int priority,
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child,
		double power):
		SingleOpExpression(priority, type, rows, cols, has_variable, has_differential, child),
		_power(power) {}

	double _power;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child, double power);
};

class Diagonalization : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

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
		int priority,
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		SingleOpExpression(priority, type, rows, cols, has_variable, has_differential, child) {}

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
		const std::string& operator_sign,
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr lhs,
		ExpressionPtr rhs):
		Expression(priority, type, rows, cols, has_variable, has_differential),
		_lhs(lhs), _rhs(rhs), _operator_sign(operator_sign) {}

	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override = 0;
	ExpressionPtr Vectorize() const override = 0;


#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	ExpressionPtr _lhs, _rhs;
	const std::string _operator_sign;
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

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr Differentiate() const override = 0;
	ExpressionPtr Vectorize() const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif
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
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

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
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

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
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(int order);

	int _order;
};

class Constant : public LeafExpression {
public:
	Constant(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		double value):
		LeafExpression(ExpressionType::kScalarConstant, rows, cols, has_variable, has_differential),
		_value(value) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr Differentiate() const override;
	ExpressionPtr Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(double value);
	double _value;
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