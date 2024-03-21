#pragma once

#include <string>

namespace TMD {
    class Expression {
    public:
        Expression(
            int rows, int cols,
            bool has_variable,
            bool has_differential):
            _rows(rows), _cols(cols),
            _has_variable(has_variable),
            _has_differential(has_differential) {}

        virtual void Print(std::ostream& out) const = 0;
        virtual Expression* Clone() const = 0;
        virtual Expression* Differentiate() = 0;
        virtual Expression* Vectorize() = 0;

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

        void Print(std::ostream &out) const override = 0;
        Expression * Clone() const override = 0;
        Expression * Differentiate() override = 0;
        Expression * Vectorize() override = 0;

        Expression *_child;
    };

    class Negative : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() override;
        Expression * Vectorize() override;

        static Negative* GetNegativeExpression(Expression* child);
    };

    class Inverse : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;
        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() override;
        Expression * Vectorize() override;

        static Inverse* GetInverseExpression(Expression* child);
    };

    class MatrixDeterminant : public SingleOpExpression {
    public:
        using SingleOpExpression::SingleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() override;
        Expression * Vectorize() override;

        static MatrixDeterminant* GetMatrixDeterminantExpression(Expression* child);
    };

    class MatrixVectorization : public SingleOpExpression {
    public:
    };

    class ScalarPower : public SingleOpExpression {
        double _power;
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

        void Print(std::ostream &out) const override = 0;
        Expression* Clone() const override = 0;
        Expression* Differentiate() override = 0;
        Expression* Vectorize() override = 0;

        Expression *_lhs, *_rhs;
    };

    class MatrixAddition : public DoubleOpExpression {

    };

    class MatrixMultiplication : public DoubleOpExpression {
    public:
        using DoubleOpExpression::DoubleOpExpression;

        void Print(std::ostream &out) const override;
        Expression * Clone() const override;
        Expression * Differentiate() override;
        Expression * Vectorize() override;

        static MatrixMultiplication* GetMatrixMultiplicationExpression(Expression* lhs, Expression* rhs);
    };

    class KroneckerProduct : public DoubleOpExpression {

    };

    class ScalarMatrixProduct : public DoubleOpExpression {

    };

    class LeafExpression : public Expression {

    };

    class IdentityMatrix : public LeafExpression {
        int _order;
    };

    class CommutationMatrix : public LeafExpression {
        int _m, _n;
    };

    class HMatrix : public LeafExpression {
    };

    class Variable : public LeafExpression {
        std::string _name;
        unsigned int _uuid;
    };
}