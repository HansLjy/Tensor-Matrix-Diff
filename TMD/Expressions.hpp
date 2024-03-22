#pragma once

#include <string>

namespace TMD {
	enum class ExpressionType {
		kNegateOp,
		kInverseOp,
		kDeterminantOp,
		kVertorizationOp,
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

		virtual ExpressionType GetExpressionType() = 0;
		virtual void MarkVariable(unsigned int uuid) = 0;

        virtual void Print(std::ostream& out) const = 0;
        virtual Expression* Clone() const = 0;

		// return a **new** expression that equals
		// the differential of the current one
        virtual Expression* Differentiate() const = 0;
        virtual Expression* Vectorize() const = 0;

        virtual ~Expression();

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
		
		void MarkVariable(unsigned int uuid) override;

        void Print(std::ostream &out) const override = 0;
        Expression * Clone() const override = 0;
        Expression * Differentiate() const override = 0;
        Expression * Vectorize() const override = 0;

        Expression *_child;
    };

    class Negate : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() const override;
        Expression * Vectorize() const override;

        static Negate* GetNegateExpression(Expression* child);
    };

    class Inverse : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;
        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() const override;
        Expression * Vectorize() const override;

        static Inverse* GetInverseExpression(Expression* child);
    };

    class Determinant : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() const override;
        Expression * Vectorize() const override;

        static Determinant* GetMatrixDeterminantExpression(Expression* child);
    };

    class Vectorization : public SingleOpExpression {
    public:
		using SingleOpExpression::SingleOpExpression;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

		static Vectorization* GetVectorizationExpression(Expression* child);
    };

	class Transpose : public SingleOpExpression {
	public:
		using SingleOpExpression::SingleOpExpression;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

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

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

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

		void MarkVariable(unsigned int uuid) override;

        void Print(std::ostream &out) const override = 0;
        Expression* Clone() const override = 0;
        Expression* Differentiate() const override = 0;
        Expression* Vectorize() const override = 0;

        Expression *_lhs, *_rhs;
    };

    class MatrixAddition : public DoubleOpExpression {
	public:
		using DoubleOpExpression::DoubleOpExpression;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

		static MatrixAddition* GetMatrixAdditionExpression(Expression* lhs, Expression* rhs);
    };

    class MatrixProduct : public DoubleOpExpression {
    public:
        using DoubleOpExpression::DoubleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() const override;
        Expression * Vectorize() const override;

        static MatrixProduct* GetMatrixProductExpression(Expression* lhs, Expression* rhs);
    };

    class KroneckerProduct : public DoubleOpExpression {
	public:
		using DoubleOpExpression::DoubleOpExpression;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

		static KroneckerProduct* GetKroneckerProduct(Expression* lhs, Expression* rhs);
    };

    class ScalarMatrixProduct : public DoubleOpExpression {
	public:
		// lhs will be the scalar
		using DoubleOpExpression::DoubleOpExpression;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

		static ScalarMatrixProduct* GetScalarMatrixProduct(Expression* scalar, Expression* matrix);
    };

    class LeafExpression : public Expression {
	public:
		using Expression::Expression;

		void MarkVariable(unsigned int uuid) override;

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;
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
		
		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

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
		
		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

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

		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

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
		
		void MarkVariable(unsigned int uuid) override;
		
		void Print(std::ostream &out) const override;
		Expression * Clone() const override;
		Expression * Differentiate() const override;
		Expression * Vectorize() const override;

		static Variable* GetVariable(const std::string& name, unsigned int uuid, int rows, int cols);

        std::string _name;
        unsigned int _uuid;
    };

	class UUIDGenerator {
	public:
		static unsigned int GenUUID();
	};
}