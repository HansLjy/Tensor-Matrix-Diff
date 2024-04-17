#include "Expressions.hpp"
#include <iostream>
#include "gtest/gtest.h"
#include "FiniteDifference.hpp"
#include "MatrixManipulation.hpp"

template<class Op>
void GetDoubleOpDerivatives(
	int A_rows, int A_cols,
	int B_rows, int B_cols,
	double step,
	TMD::ExpressionPtr& AxB,
	TMD::ExpressionPtr& gradient_A,
	TMD::ExpressionPtr& gradient_B,
	Eigen::MatrixXd& numeric_gradient_A,
	Eigen::MatrixXd& analytic_gradient_A,
	Eigen::MatrixXd& numeric_gradient_B,
	Eigen::MatrixXd& analytic_gradient_B
) {
	int variable_id = 0;
	auto A = TMD::Variable::GetSelfType("A", variable_id++, A_rows, A_cols);
	auto B = TMD::Variable::GetSelfType("B", variable_id++, B_rows, B_cols);
	AxB = Op::GetSelfType(A, B);
	gradient_A = TMD::GetDerivative(AxB, A->_variable_id);
	gradient_B = TMD::GetDerivative(AxB, B->_variable_id);
	
	TMD::VariableTable table;
	
	Eigen::MatrixXd A_mat = Eigen::MatrixXd::Random(A_rows, A_cols);
	Eigen::MatrixXd B_mat = Eigen::MatrixXd::Random(B_rows, B_cols);
	
	table[A->_variable_id] = A_mat;
	table[B->_variable_id] = B_mat;
	
	auto tmp_table = table;
	
	auto func_A = [&tmp_table, &A, &AxB] (const Eigen::MatrixXd& A_mat) -> Eigen::MatrixXd {
		tmp_table[A->_variable_id] = A_mat;
		return AxB->SlowEvaluation(tmp_table);
	};
	
	auto func_B = [&tmp_table, &B, &AxB] (const Eigen::MatrixXd& B_mat) -> Eigen::MatrixXd {
		tmp_table[B->_variable_id] = B_mat;
		return AxB->SlowEvaluation(tmp_table);
	};
	
	numeric_gradient_A = Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func_A, A_mat, step);
	analytic_gradient_A = gradient_A->SlowEvaluation(table);
	
	tmp_table = table;
	numeric_gradient_B = Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func_B, B_mat, step);
	analytic_gradient_B = gradient_B->SlowEvaluation(table);
}

TEST(DerivativeTest, MatrixAdditionTest) {
	TMD::ExpressionPtr AxB, gradient_A, gradient_B;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDoubleOpDerivatives<TMD::MatrixAddition>(
		3, 4,
		3, 4,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);
}

TEST(DerivativeTest, MatrixProductTest) {
	TMD::ExpressionPtr AxB, gradient_A, gradient_B;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDoubleOpDerivatives<TMD::MatrixProduct>(
		3, 4,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);
}

TEST(DerivativeTest, KroneckerProductTest) {
	TMD::ExpressionPtr AxB, gradient_A, gradient_B;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDoubleOpDerivatives<TMD::KroneckerProduct>(
		3, 4,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);
}

TEST(DerivativeTest, ScalarMatrixProductTest) {
	TMD::ExpressionPtr AxB, gradient_A, gradient_B;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDoubleOpDerivatives<TMD::ScalarMatrixProduct>(
		1, 1,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);
}

TEST(DerivativeTest, HadamardProductTest) {
	TMD::ExpressionPtr AxB, gradient_A, gradient_B;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDoubleOpDerivatives<TMD::HadamardProduct>(
		4, 5,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);
}

template<class Op>
void GetSingleOpDerivative(
	int A_rows, int A_cols,
	double step,
	TMD::ExpressionPtr& op_A,
	TMD::ExpressionPtr& gradient_A,
	Eigen::MatrixXd& numeric_gradient_A,
	Eigen::MatrixXd& analytic_gradient_A
) {
	int variable_id = 0;
	auto A = TMD::Variable::GetSelfType("A", variable_id++, A_rows, A_cols);
	op_A = Op::GetSelfType(A);
	gradient_A = TMD::GetDerivative(op_A, A->_variable_id);
	
	Eigen::MatrixXd A_mat = Eigen::MatrixXd::Random(A_rows, A_cols);
	
	TMD::VariableTable table;
	table[A->_variable_id] = A_mat;
	
	auto func = [&table, &A, &op_A] (const Eigen::MatrixXd& A_mat) -> Eigen::MatrixXd {
		table[A->_variable_id] = A_mat;
		return op_A->SlowEvaluation(table);
	};
	
	numeric_gradient_A = Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func, A_mat, step);
	analytic_gradient_A = gradient_A->SlowEvaluation(table);
}

TEST(DerivativeTest, PowerTest) {
	int variable_id = 0;
	auto A = TMD::Variable::GetSelfType("A", variable_id++, 4, 5);
	auto B = TMD::RationalScalarConstant::GetSelfType(TMD::RationalScalarConstant::RationalNumber(1, 2));
	auto AxB = TMD::MatrixScalarPower::GetSelfType(A, B);
	
	auto gradient = TMD::GetDerivative(AxB, A->_variable_id);
	
	TMD::VariableTable table;
	
	Eigen::MatrixXd A_mat = Eigen::MatrixXd::Random(4, 5).cwiseAbs();
	
	table[A->_variable_id] = A_mat;
	
	auto tmp_table = table;
	auto func = [&tmp_table, &A, &AxB] (const Eigen::MatrixXd& A_mat) -> Eigen::MatrixXd {
		tmp_table[A->_variable_id] = A_mat;
		return AxB->SlowEvaluation(tmp_table);
	};
	
	auto numeric_gradient = Numerics::MatrixGradient<Eigen::MatrixXd, Eigen::MatrixXd>(func, A_mat, 1e-8);
	auto analytic_gradient = gradient->SlowEvaluation(table);

	EXPECT_LT((numeric_gradient - analytic_gradient).norm(), 1e-4);
}

TEST(DerivativeTest, NegateTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Negate>(
		3, 4,
		1e-4,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, InverseTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Inverse>(
		4, 4,
		1e-7,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).lpNorm<Eigen::Infinity>(), 1e-4);
}

TEST(DerivativeTest, DeterminantTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Determinant>(
		4, 4,
		1e-7,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, VectorizationTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Vectorization>(
		3, 4,
		1e-4,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, TransposeTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Transpose>(
		3, 4,
		1e-4,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, SkewTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Skew>(
		3, 1,
		1e-4,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, ExpTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Exp>(
		3, 4,
		1e-8,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(DerivativeTest, DiagonalizeTest) {
	TMD::ExpressionPtr op_A, gradient_A;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A;
	GetSingleOpDerivative<TMD::Diagonalization>(
		4, 1,
		1e-4,
		op_A, gradient_A,
		numeric_gradient_A, analytic_gradient_A
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
}

TEST(FunctionalityTest, SubstitutionTest) {
	int variable_id = 0;
	auto A = TMD::Variable::GetSelfType("A", variable_id++, 4, 3);
	auto B = TMD::Variable::GetSelfType("B", variable_id++, 3, 3);
	auto a = TMD::Variable::GetSelfType("a", variable_id++, 4, 1);
	auto b = TMD::Variable::GetSelfType("b", variable_id++, 3, 1);
	
	auto X = TMD::Variable::GetSelfType("X", variable_id++, 3, 1);
	auto Y = TMD::Variable::GetSelfType("Y", variable_id++, 3, 1);
	
	auto AXpa = TMD::MatrixAddition::GetSelfType(
		TMD::MatrixProduct::GetSelfType(A, X), a
	);

	auto BYpb = TMD::MatrixAddition::GetSelfType(
		TMD::MatrixProduct::GetSelfType(B, Y), b
	);

	auto subed = AXpa->Substitute({{X->_variable_id, BYpb}});
	std::cerr << *subed << std::endl;

	Eigen::MatrixXd A_mat = Eigen::MatrixXd::Random(4, 3);
	Eigen::MatrixXd B_mat = Eigen::MatrixXd::Random(3, 3);
	Eigen::MatrixXd a_mat = Eigen::MatrixXd::Random(4, 1);
	Eigen::MatrixXd b_mat = Eigen::MatrixXd::Random(3, 1);

	Eigen::MatrixXd X_mat = Eigen::MatrixXd::Random(3, 1);
	Eigen::MatrixXd Y_mat = Eigen::MatrixXd::Random(3, 1);

	TMD::VariableTable table;
	table[A->_variable_id] = A_mat;
	table[B->_variable_id] = B_mat;
	table[a->_variable_id] = a_mat;
	table[b->_variable_id] = b_mat;
	table[X->_variable_id] = X_mat;
	table[Y->_variable_id] = Y_mat;

	auto result = subed->SlowEvaluation(table);
	Eigen::MatrixXd expected = A_mat * (B_mat * Y_mat + b_mat) + a_mat;

	EXPECT_LT((result - expected).norm(), 1e-5);
}

TEST(FunctionalityTest, COWTest) {
	int variable_id = 0;
	auto x = TMD::Variable::GetSelfType("x", variable_id++, 2, 1);
	auto p = TMD::Variable::GetSelfType("p", variable_id++, 2, 1);
	auto V = TMD::Variable::GetSelfType("V", variable_id++, 2, 2);

	auto x_minus_p = TMD::MatrixAddition::GetSelfType(x, TMD::Negate::GetSelfType(p));

	auto exp_part = TMD::Exp::GetSelfType(
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::RationalScalarConstant::GetSelfType(TMD::RationalScalarConstant::RationalNumber(-1, 2)),
			TMD::GetMultipleProduct({
				TMD::Transpose::GetSelfType(x_minus_p),
				TMD::Inverse::GetSelfType(V),
				x_minus_p
			})
		)
	);

	auto mult_part = TMD::MatrixScalarPower::GetSelfType(
		TMD::Determinant::GetSelfType(V),
		TMD::RationalScalarConstant::GetSelfType(
			TMD::RationalScalarConstant::RationalNumber(-1, 2)
		)
	);

	auto gaussian = TMD::MatrixProduct::GetSelfType(mult_part, exp_part);

	std::stringstream strstrm;
	gaussian->Print(strstrm);
	std::string str_before = strstrm.str();

	auto gaussian_derivative = TMD::GetDerivative(gaussian, V->_variable_id);
	
	strstrm.str(std::string());
	gaussian->Print(strstrm);
	std::string str_after = strstrm.str();

	EXPECT_EQ(str_before, str_after);
}