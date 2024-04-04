#pragma once

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
Expression* GetDerivative(const Expression* expression, unsigned int variable);

enum class ExpressionType {
	kNegateOp,
	kInverseOp,
	kDeterminantOp,
	kVectorizationOp,
	kTransposeOp,
	kScalarPowerOp,
	kMatrixAdditionOp,
	kMatrixProductOp,
	kKroneckerProductOp,
	kScalarMatrixProductOp,
	kIdentityMatrix,
	kCommutationMatrix,
	kScalarConstant,
	kVariable
};

class Expression {
public:
	Expression(
		int rows, int cols,
		bool has_variable,
		bool has_differential):
		_rows(rows), _cols(cols),
		_has_variable(has_variable),
		_has_differential(has_differential) {}

	virtual ExpressionType GetExpressionType() const = 0;
	virtual void MarkVariable(unsigned int uuid) = 0;
	virtual void MarkDifferential() = 0;

	virtual Expression* GetTransposedDerivative() const;

	virtual void Print(std::ostream& out) const = 0;
	virtual Expression* Clone() const = 0;

	// return a **new** expression that equals
	// the differential of the current one
	virtual Expression* Differentiate() const = 0;
	virtual Expression* Vectorize() const = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	virtual Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const = 0;
#endif

	virtual ~Expression() = default;

	int _rows, _cols;           // for scalar expression, simply set these to (1, 1)
	bool _has_variable;
	bool _has_differential;
};

class SingleOpExpression : public Expression {
public:
	SingleOpExpression(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		Expression* child):
		Expression(rows, cols, has_variable, has_differential),
		_child(child) {}
	
	ExpressionType GetExpressionType() const override = 0;
	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;

	void Print(std::ostream &out) const override = 0;
	Expression * Clone() const override = 0;
	Expression * Differentiate() const override = 0;
	Expression * Vectorize() const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	~SingleOpExpression() override;

	Expression *_child;
};

class Negate : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Negate* GetNegateExpression(Expression* child);
};

class Inverse : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Inverse* GetInverseExpression(Expression* child);
};

class Determinant : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Determinant* GetMatrixDeterminantExpression(Expression* child);
};

class Vectorization : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Vectorization* GetVectorizationExpression(Expression* child);
};

class Transpose : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Transpose* GetTransposeExpression(Expression* child);
};

class ScalarPower : public SingleOpExpression {
public:
	ScalarPower(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		Expression* child,
		double power):
		SingleOpExpression(rows, cols, has_variable, has_differential, child),
		_power(power) {}

	double _power;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ScalarPower* GetScalarPowerExpression(Expression* child, double power);
};

class DoubleOpExpression : public Expression {
public:
	DoubleOpExpression(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		Expression* lhs,
		Expression* rhs):
		Expression(rows, cols, has_variable, has_differential),
		_lhs(lhs), _rhs(rhs) {}

	ExpressionType GetExpressionType() const override = 0;
	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;

	void Print(std::ostream &out) const override = 0;
	Expression* Clone() const override = 0;
	Expression* Differentiate() const override = 0;
	Expression* Vectorize() const override = 0;


#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	~DoubleOpExpression();

	Expression *_lhs, *_rhs;
};

class MatrixAddition : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionType GetExpressionType() const override;

	Expression* GetTransposedDerivative() const override;

	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static MatrixAddition* GetSelfType(Expression* lhs, Expression* rhs);
};

class MatrixProduct : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionType GetExpressionType() const override;
	Expression* GetTransposedDerivative() const override;

	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static MatrixProduct* GetSelfType(Expression* lhs, Expression* rhs);
};

class KroneckerProduct : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static KroneckerProduct* GetSelfType(Expression* lhs, Expression* rhs);
};

class ScalarMatrixProduct : public DoubleOpExpression {
public:
	// lhs will be the scalar
	using DoubleOpExpression::DoubleOpExpression;

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ScalarMatrixProduct* GetSelfType(Expression* scalar, Expression* matrix);
};

class LeafExpression : public Expression {
public:
	using Expression::Expression;

	ExpressionType GetExpressionType() const override = 0;
	void MarkVariable(unsigned int uuid) override;
	void MarkDifferential() override;

	void Print(std::ostream &out) const override = 0;
	Expression * Clone() const override = 0;
	Expression * Differentiate() const override = 0;
	Expression * Vectorize() const override = 0;

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
		LeafExpression(rows, cols, has_variable, has_differential),
		_order(order) {}
	
	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static IdentityMatrix* GetIdentityMatrix(int order);

	int _order;
};

class CommutationMatrix : public LeafExpression {
public:
	CommutationMatrix(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		int m, int n):
		LeafExpression(rows, cols, has_variable, has_differential),
		_m(m), _n(n) {}
	
	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Expression* GetCommutationMatrix(int m, int n);

	int _m, _n;
};

class ScalarConstant : public LeafExpression {
public:
	ScalarConstant(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		double value):
		LeafExpression(rows, cols, has_variable, has_differential),
		_value(value) {}

	ExpressionType GetExpressionType() const override;
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ScalarConstant* GetScalarConstant(double value);
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
		LeafExpression(rows, cols, has_variable, has_differential),
		_name(name), _uuid(uuid) {}

	ExpressionType GetExpressionType() const override;
	void MarkVariable(unsigned int uuid) override;

	Expression* GetTransposedDerivative() const override;
	
	void Print(std::ostream &out) const override;
	Expression * Clone() const override;
	Expression * Differentiate() const override;
	Expression * Vectorize() const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static Variable* GetVariable(const std::string& name, unsigned int uuid, int rows, int cols);

	std::string _name;
	unsigned int _uuid;
};

class UUIDGenerator {
public:
	static unsigned int _cnt;
	static unsigned int GenUUID();
};

}