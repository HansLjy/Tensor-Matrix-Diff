#include "Expressions.hpp"
#include <iostream>
#include "gtest/gtest.h"
#include "FiniteDifference.hpp"

template<class Op>
void GetDerivatives(
	int A_rows, int A_cols,
	int B_rows, int B_cols,
	double step,
	TMD::Expression*& AxB,
	TMD::Expression*& gradient_A,
	TMD::Expression*& gradient_B,
	Eigen::MatrixXd& numeric_gradient_A,
	Eigen::MatrixXd& analytic_gradient_A,
	Eigen::MatrixXd& numeric_gradient_B,
	Eigen::MatrixXd& analytic_gradient_B
) {
	auto A = TMD::Variable::GetVariable("A", TMD::UUIDGenerator::GenUUID(), A_rows, A_cols);
	auto B = TMD::Variable::GetVariable("B", TMD::UUIDGenerator::GenUUID(), B_rows, B_cols);
	AxB = Op::GetSelfType(A, B);
	gradient_A = TMD::GetDerivative(AxB, A->_uuid);
	gradient_B = TMD::GetDerivative(AxB, B->_uuid);
	
	TMD::VariableTable table;
	
	Eigen::MatrixXd A_mat = Eigen::MatrixXd::Random(A_rows, A_cols);
	Eigen::MatrixXd B_mat = Eigen::MatrixXd::Random(B_rows, B_cols);
	
	table[A->_uuid] = A_mat;
	table[B->_uuid] = B_mat;
	
	auto tmp_table = table;
	
	auto func_A = [&tmp_table, &A, &AxB] (const Eigen::MatrixXd& A_mat) -> Eigen::MatrixXd {
		tmp_table[A->_uuid] = A_mat;
		return AxB->SlowEvaluation(tmp_table);
	};
	
	auto func_B = [&tmp_table, &B, &AxB] (const Eigen::MatrixXd& B_mat) -> Eigen::MatrixXd {
		tmp_table[B->_uuid] = B_mat;
		return AxB->SlowEvaluation(tmp_table);
	};
	
	numeric_gradient_A = Numerics::Gradient<Eigen::MatrixXd, Eigen::MatrixXd>(func_A, A_mat, step);
	analytic_gradient_A = gradient_A->SlowEvaluation(table);
	
	tmp_table = table;
	numeric_gradient_B = Numerics::Gradient<Eigen::MatrixXd, Eigen::MatrixXd>(func_B, B_mat, step);
	analytic_gradient_B = gradient_B->SlowEvaluation(table);
}

TEST(DerivativeTest, MatrixAdditionTest) {
	TMD::Expression* AxB = nullptr, *gradient_A = nullptr, *gradient_B = nullptr;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDerivatives<TMD::MatrixAddition>(
		3, 4,
		3, 4,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);

	delete AxB;
	delete gradient_A;
	delete gradient_B;
}

TEST(DerivativeTest, MatrixProductTest) {
	TMD::Expression* AxB = nullptr, *gradient_A = nullptr, *gradient_B = nullptr;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDerivatives<TMD::MatrixProduct>(
		3, 4,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);

	delete AxB;
	delete gradient_A;
	delete gradient_B;
}

TEST(DerivativeTest, KroneckerProductTest) {
	TMD::Expression* AxB = nullptr, *gradient_A = nullptr, *gradient_B = nullptr;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDerivatives<TMD::KroneckerProduct>(
		3, 4,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);

	delete AxB;
	delete gradient_A;
	delete gradient_B;
}

TEST(DerivativeTest, ScalarMatrixProductTest) {
	TMD::Expression* AxB = nullptr, *gradient_A = nullptr, *gradient_B = nullptr;
	Eigen::MatrixXd numeric_gradient_A, analytic_gradient_A, numeric_gradient_B, analytic_gradient_B;
	GetDerivatives<TMD::ScalarMatrixProduct>(
		1, 1,
		4, 5,
		1e-4,
		AxB, gradient_A, gradient_B,
		numeric_gradient_A, analytic_gradient_A,
		numeric_gradient_B, analytic_gradient_B
	);
	EXPECT_LT((numeric_gradient_A - analytic_gradient_A).norm(), 1e-4);
	EXPECT_LT((numeric_gradient_B - analytic_gradient_B).norm(), 1e-4);

	delete AxB;
	delete gradient_A;
	delete gradient_B;
}