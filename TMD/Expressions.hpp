#pragma once

#include <memory>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <filesystem>
#include "TMDConfig.hpp"
#ifdef IMPLEMENT_SLOW_EVALUATION
	#include "Eigen/Eigen"
#endif

namespace fs = std::filesystem;

namespace TMD {

class Expression;
namespace internal {
	class Variable;
}

typedef std::shared_ptr<Expression> ExpressionPtr;
typedef std::shared_ptr<const Expression> ConstExpressionPtr;
typedef std::shared_ptr<internal::Variable> VariablePtr;
typedef std::shared_ptr<const internal::Variable> ConstVariablePtr;

#ifdef IMPLEMENT_SLOW_EVALUATION

typedef std::map<unsigned int, Eigen::MatrixXd> VariableTable;

Eigen::MatrixXd GetExpressionNumericDerivative(
	const ExpressionPtr expression,
	const VariableTable& table,
	unsigned int variable_id
);

#endif

// single op
ExpressionPtr GetNegate(const ExpressionPtr expr);
ExpressionPtr GetInverse(const ExpressionPtr expr);
ExpressionPtr GetDeterminant(const ExpressionPtr expr);
ExpressionPtr GetVectorization(const ExpressionPtr expr);
ExpressionPtr GetTranspose(const ExpressionPtr expr);
ExpressionPtr GetDiagonalization(const ExpressionPtr expr);
ExpressionPtr GetSkew(const ExpressionPtr expr);
ExpressionPtr GetExp(const ExpressionPtr expr);

ExpressionPtr GetVector2Norm(const ExpressionPtr x);

// double op
ExpressionPtr GetAddition(const std::vector<ExpressionPtr>& adds);
ExpressionPtr GetProduct(const std::vector<ExpressionPtr>& prods);
ExpressionPtr GetKroneckerProduct(const std::vector<ExpressionPtr>& prods);
ExpressionPtr GetScalarProduct(const ExpressionPtr scalar, const ExpressionPtr matrix);
ExpressionPtr GetPower(const ExpressionPtr matrix, const ExpressionPtr power);
ExpressionPtr GetHadamardProduct(const ExpressionPtr lhs, const ExpressionPtr rhs);

ExpressionPtr GetMinus(const ExpressionPtr lhs, const ExpressionPtr rhs);
ExpressionPtr GetDotProduct(const ExpressionPtr lhs, const ExpressionPtr rhs);

// leaf
VariablePtr GetVariable(const std::string& name, unsigned int variable_id, int rows, int cols);
ExpressionPtr GetIdeneityMatrix(int order);
ExpressionPtr GetCommutationMatrix(int m, int n);
ExpressionPtr GetDiagonalizationMatrix(int order);
ExpressionPtr GetSkewMatrix();
ExpressionPtr GetSelectMatrix(int begin, int end, int total);
ExpressionPtr GetRationalScalarConstant(int n);
ExpressionPtr GetRationalScalarConstant(int p, int q);

// taking derivatives
ExpressionPtr GetDerivative(const ExpressionPtr expression, unsigned int variable);

// exporting expression
void ExportGraph(const ExpressionPtr expression, const fs::path& filepath);

class Expression : public std::enable_shared_from_this<Expression> {
public:
	ExpressionPtr MarkVariable(unsigned int variable_id) const;
	ExpressionPtr Differentiate();
	ExpressionPtr MarkDifferential();
	ExpressionPtr Vectorize() const;
	ExpressionPtr Substitute(const std::map<unsigned int, ExpressionPtr>& subs);
	ExpressionPtr Simplify();
	std::string ExportGraph() const;
	virtual ExpressionPtr GetTransposedDerivative() const;

	virtual void Print(std::ostream& out) const = 0;
	virtual ExpressionPtr Clone() const = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	virtual Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const = 0;
#endif

	virtual ~Expression() = default;

public:
	ExpressionPtr RealMarkVariable(unsigned int variable_id, std::map<unsigned int, ExpressionPtr>& marked_exprs) const;
	virtual ExpressionPtr GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr>& marked_exprs) const = 0;

	ExpressionPtr RealDifferentiate(std::map<unsigned int, ExpressionPtr>& diffed_exprs);
	virtual ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr>& diffed_exprs) = 0;

	ExpressionPtr RealMarkDifferential(std::map<unsigned int, ExpressionPtr>& marked_exprs);
	virtual ExpressionPtr GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr>& marked_exprs) = 0;

	ExpressionPtr RealVectorize(std::map<unsigned int, ExpressionPtr>& veced_exprs) const;
	virtual ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr>& veced_exprs) const = 0;

	ExpressionPtr RealSubstitute(const std::map<unsigned int, ExpressionPtr>& subs, std::map<unsigned int, ExpressionPtr>& subed_exprs);
	virtual ExpressionPtr GetSubedExpression(const std::map<unsigned int, ExpressionPtr>& subs, std::map<unsigned int, ExpressionPtr>& subed_exprs) = 0;

	ExpressionPtr RealSimplify(std::map<unsigned int, ExpressionPtr>& simplified_exprs);
	virtual ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr>& simplified_exprs) = 0;

	std::map<unsigned int, int> CountInDegree() const;
	virtual void RealCountInDegree(
		std::set<unsigned int>& visited,
		std::map<unsigned int, int>& in_degrees
	) const = 0;
	void RealExportGraph(
		int& tree_cnt,
		int cur_tree_id,
		int &tree_node_cnt,
		const std::map<unsigned int, unsigned int>& duplicated_expr_ids,
		std::vector<bool>& duplicated_expr_exported,
		std::vector<std::string>& labels,
		std::stringstream& out
	) const;
	virtual void GetExportedGraph(
		int& tree_cnt,
		int cur_tree_id,
		int& tree_node_cnt,
		const std::map<unsigned int, unsigned int>& duplicated_expr_ids,
		std::vector<bool>& duplicated_expr_exported,
		std::vector<std::string>& labels,
		std::stringstream& out
	) const = 0;

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
		kSelectMatrix,
		kElementMatrix,
		kRationalScalarConstant,
		kVariable
	};

public:
	const unsigned int _uuid;
	const int _priority;
	const ExpressionType _type;

	int _rows, _cols;           // for scalar expression, simply set these to (1, 1)
	bool _has_variable;
	bool _has_differential;

protected:
	Expression(
		int priority,
		const ExpressionType& type,
		int rows, int cols,
		bool has_variable,
		bool has_differential
	);
	
};

namespace internal {

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

class SingleOpExpression : public Expression {
public:
	ExpressionPtr GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr> &marked_exprs) const override;
	ExpressionPtr GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr> &marked_exprs) override;
	ExpressionPtr GetSubedExpression(const std::map<unsigned int, ExpressionPtr> &subs, std::map<unsigned int, ExpressionPtr> &subed_exprs) override;

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override = 0;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override = 0;

	void GetExportedGraph(int &tree_cnt, int cur_tree_id, int &tree_node_cnt, const std::map<unsigned int, unsigned int> &duplicated_expr_ids, std::vector<bool> &duplicated_expr_exported, std::vector<std::string> &labels, std::stringstream &out) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;
	void RealCountInDegree(std::set<unsigned int> &visited, std::map<unsigned int, int> &in_degrees) const override;

	virtual ExpressionPtr GetSingleOpExpression(ExpressionPtr child) const = 0;

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
	
};

template<class Derived>
class ConcreteSingleOpExpression : public SingleOpExpression {
public:
	using SingleOpExpression::SingleOpExpression;
	ExpressionPtr GetSingleOpExpression(ExpressionPtr child) const override {
		return Derived::GetSelfType(child);
	}

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override = 0;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif
};

class Negate : public ConcreteSingleOpExpression<Negate> {
public:
	Negate(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Negate>(kNegatePriority, ExpressionType::kNegateOp, "-", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Inverse : public ConcreteSingleOpExpression<Inverse> {
public:
	Inverse(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Inverse>(kInversePriority, ExpressionType::kInverseOp, "\\text{inv}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Determinant : public ConcreteSingleOpExpression<Determinant> {
public:
	Determinant(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Determinant>(kDeterminantPriority, ExpressionType::kDeterminantOp, "\\det", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Vectorization : public ConcreteSingleOpExpression<Vectorization> {
public:
	Vectorization(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Vectorization> (kVectorizationPriority, ExpressionType::kDeterminantOp, "\\text{vec}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Transpose : public ConcreteSingleOpExpression<Transpose> {
public:
	Transpose(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Transpose> (kTransposePriority, ExpressionType::kTransposeOp, "\\top", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Diagonalization : public ConcreteSingleOpExpression<Diagonalization> {
public:
	Diagonalization(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Diagonalization>(kDiagonalizePriority, ExpressionType::kDiagonalizeOp, "\\text{diag}", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Skew : public ConcreteSingleOpExpression<Skew> {
public:
	Skew(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Skew> (
			kSkewPriority, ExpressionType::kSkewOp,
			"\\left[\\right]",
			rows, cols,
			has_variable, has_differential,
			child) {}
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr child);
};

class Exp : public ConcreteSingleOpExpression<Exp> {
public:
	Exp(
		int rows, int cols,
		bool has_variable,
		bool has_differential,
		ExpressionPtr child):
		ConcreteSingleOpExpression<Exp> (kExpPriority, ExpressionType::kExpOp, "\\exp", rows, cols, has_variable, has_differential, child) {}

	void Print(std::ostream& out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

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

	ExpressionPtr GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr> &marked_exprs) const override;
	ExpressionPtr GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr> &marked_exprs) override;
	ExpressionPtr GetSubedExpression(const std::map<unsigned int, ExpressionPtr> &subs, std::map<unsigned int, ExpressionPtr> &subed_exprs) override;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override = 0;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override = 0;

	void GetExportedGraph(int &tree_cnt, int cur_tree_id, int &tree_node_cnt, const std::map<unsigned int, unsigned int> &duplicated_expr_ids, std::vector<bool> &duplicated_expr_exported, std::vector<std::string> &labels, std::stringstream &out) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;
	void RealCountInDegree(std::set<unsigned int> &visited, std::map<unsigned int, int> &in_degrees) const override;
	
	virtual ExpressionPtr GetDoubleOpExpression(ExpressionPtr lhs, ExpressionPtr rhs) const = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif

	ExpressionPtr _lhs, _rhs;
	const std::string _operator_sym;
};

template<class Derived>
class ConcreteDoubleOpExpression : public DoubleOpExpression {
public:
	using DoubleOpExpression::DoubleOpExpression;
	ExpressionPtr GetDoubleOpExpression(ExpressionPtr lhs, ExpressionPtr rhs) const override {
		return Derived::GetSelfType(lhs, rhs);
	}

	ExpressionPtr Clone() const override = 0;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override = 0;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override = 0;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override = 0;
#endif
};

class MatrixAddition : public ConcreteDoubleOpExpression<MatrixAddition> {
public:
	using ConcreteDoubleOpExpression<MatrixAddition>::ConcreteDoubleOpExpression;

	ExpressionPtr GetTransposedDerivative() const override;

	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class MatrixProduct : public ConcreteDoubleOpExpression<MatrixProduct> {
public:
	using ConcreteDoubleOpExpression<MatrixProduct>::ConcreteDoubleOpExpression;

	ExpressionPtr GetTransposedDerivative() const override;

	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class KroneckerProduct : public ConcreteDoubleOpExpression<KroneckerProduct> {
public:
	using ConcreteDoubleOpExpression<KroneckerProduct>::ConcreteDoubleOpExpression;

	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr lhs, ExpressionPtr rhs);
};

class ScalarMatrixProduct : public ConcreteDoubleOpExpression<ScalarMatrixProduct> {
public:
	// lhs will be the scalar
	using ConcreteDoubleOpExpression<ScalarMatrixProduct>::ConcreteDoubleOpExpression;

	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr scalar, ExpressionPtr matrix);
};

// WARNING: for now, the power has to be constant
class MatrixScalarPower : public ConcreteDoubleOpExpression<MatrixScalarPower> {
public:
	using ConcreteDoubleOpExpression<MatrixScalarPower>::ConcreteDoubleOpExpression;

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(ExpressionPtr matrix, ExpressionPtr power);
};

class HadamardProduct : public ConcreteDoubleOpExpression<HadamardProduct> {
public:
	using ConcreteDoubleOpExpression<HadamardProduct>::ConcreteDoubleOpExpression;
	
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

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

	ExpressionPtr GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr> &marked_exprs) const override;
	ExpressionPtr GetMarkedDifferentialExpression(std::map<unsigned int, ExpressionPtr> &marked_exprs) override;
	ExpressionPtr GetSubedExpression(const std::map<unsigned int, ExpressionPtr> &subs, std::map<unsigned int, ExpressionPtr> &subed_exprs) override;

	void Print(std::ostream &out) const override = 0;
	ExpressionPtr Clone() const override = 0;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

	void GetExportedGraph(int &tree_cnt, int cur_tree_id, int &tree_node_cnt, const std::map<unsigned int, unsigned int> &duplicated_expr_ids, std::vector<bool> &duplicated_expr_exported, std::vector<std::string> &labels, std::stringstream &out) const override;

	ExpressionPtr GetSimplifedExpression(std::map<unsigned int, ExpressionPtr> &simplified_exprs) override;
	void RealCountInDegree(std::set<unsigned int> &visited, std::map<unsigned int, int> &in_degrees) const override;

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

class SelectMatrix : public LeafExpression {
public:
	SelectMatrix(int begin, int end, int total):
		LeafExpression(ExpressionType::kSelectMatrix, end - begin, total, false, false),
		_begin(begin), _end(end), _total(total) {}

	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;

	int _begin, _end, _total;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable &table) const override;
#endif

	static ExpressionPtr GetSelfType(int begin, int end, int total);
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
		unsigned int variable_id):
		LeafExpression(ExpressionType::kVariable, rows, cols, has_variable, has_differential),
		_name(name), _variable_id(variable_id) {}

	ExpressionPtr GetMarkedVariableExpression(unsigned int variable_id, std::map<unsigned int, ExpressionPtr> &marked_exprs) const override;
	ExpressionPtr GetSubedExpression(const std::map<unsigned int, ExpressionPtr> &subs, std::map<unsigned int, ExpressionPtr> &subed_exprs) override;

	ExpressionPtr GetTransposedDerivative() const override;
	
	void Print(std::ostream &out) const override;
	ExpressionPtr Clone() const override;
	ExpressionPtr GetDiffedExpression(std::map<unsigned int, ExpressionPtr> &diffed_exprs) override;
	ExpressionPtr GetVecedExpression(std::map<unsigned int, ExpressionPtr> &veced_exprs) const override;

#ifdef IMPLEMENT_SLOW_EVALUATION
	Eigen::MatrixXd SlowEvaluation(const VariableTable& table) const override;
	Eigen::MatrixXd GetRandomMatrix() const;
#endif

	static std::shared_ptr<Variable> GetSelfType(const std::string& name, unsigned int variable_id, int rows, int cols);

	std::string _name;
	unsigned int _variable_id;
};

std::ostream& operator<<(std::ostream& out, const Expression& expr);

class UUIDGenerator {
public:
	static unsigned int _cnt;
	static unsigned int GenUUID();
};

}

}