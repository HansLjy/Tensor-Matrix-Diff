#define IMPLEMENT_SLOW_EVALUATION
#include "Expressions.hpp"
#include "FiniteDifference.hpp"
#include <iostream>

int main() {
	auto V = TMD::Variable::GetSelfType("V", TMD::UUIDGenerator::GenUUID(), 3, 3);
	auto x = TMD::Variable::GetSelfType("x", TMD::UUIDGenerator::GenUUID(), 3, 1);
	auto p = TMD::Variable::GetSelfType("p", TMD::UUIDGenerator::GenUUID(), 3, 1);

	auto x_minus_p = TMD::MatrixAddition::GetSelfType(x, TMD::Negate::GetSelfType(p));

	auto exp_part = TMD::Exp::GetSelfType(
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::Constant::GetSelfType(-0.5),
			TMD::MatrixProduct::GetSelfType(
				TMD::Transpose::GetSelfType(x_minus_p),
				TMD::MatrixProduct::GetSelfType(
					TMD::Inverse::GetSelfType(V),
					x_minus_p
				)
			)
		)
	);

	auto mult_part = TMD::ScalarPower::GetSelfType(TMD::Determinant::GetSelfType(V), -0.5);

	auto gaussian = TMD::MatrixProduct::GetSelfType(mult_part, exp_part);

	auto derivative_gaussian = TMD::GetDerivative(gaussian, V->_uuid);

	Eigen::MatrixXd V_matrix = Eigen::MatrixXd::Random(3, 3);
	Eigen::MatrixXd x_matrix = Eigen::MatrixXd::Random(3, 1);
	Eigen::MatrixXd p_matrix = Eigen::MatrixXd::Random(3, 1);

	TMD::VariableTable table;
	table[V->_uuid] = V_matrix;
	table[x->_uuid] = x_matrix;
	table[p->_uuid] = p_matrix;

	auto numeric_gradient = GetExpressionNumericDerivative(gaussian, table, V->_uuid);
	auto analytic_gradient = derivative_gaussian->SlowEvaluation(table);

	std::cerr << (numeric_gradient - analytic_gradient).norm() << std::endl;

	std::cerr << *gaussian << std::endl;
	std::cerr << *derivative_gaussian << std::endl;
}