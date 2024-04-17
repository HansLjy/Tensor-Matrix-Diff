#define IMPLEMENT_SLOW_EVALUATION
#include "Expressions.hpp"
#include "FiniteDifference.hpp"
#include "TestConfig.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
	int variable_id = 0;
	
	// camera transformation
	auto A = TMD::Variable::GetSelfType("A", variable_id++, 3, 3);
	auto b = TMD::Variable::GetSelfType("b", variable_id++, 3, 1);

	// Gaussian configuration
	auto q = TMD::Variable::GetSelfType("q", variable_id++, 4, 1);
	auto S = TMD::Variable::GetSelfType("S", variable_id++, 3, 3);
	auto p = TMD::Variable::GetSelfType("p", variable_id++, 3, 1);

	// Other constant
	auto gaussian_normalizer = TMD::Variable::GetSelfType("\\frac{1}{2 \\pi}", variable_id++, 1, 1);

	// Selection Matrices
	auto Sv = TMD::Variable::GetSelfType("\\mathcal{S}_v", variable_id++, 3, 4);
	auto Sq0 = TMD::Variable::GetSelfType("\\mathcal{S}_{q_0}", variable_id++, 1, 4);
	auto Sx1 = TMD::Variable::GetSelfType("\\mathcal{S}_{x_1}", variable_id++, 1, 3);
	auto Sx2 = TMD::Variable::GetSelfType("\\mathcal{S}_{x_2}", variable_id++, 1, 3);
	auto Sx3 = TMD::Variable::GetSelfType("\\mathcal{S}_{x_3}", variable_id++, 1, 3);
	auto S2d = TMD::Variable::GetSelfType("\\mathcal{S}_{2d}", variable_id++, 2, 3);

	// R related
	auto v = TMD::MatrixProduct::GetSelfType(Sv, q);
	auto skew_v = TMD::Skew::GetSelfType(v);
	auto q0 = TMD::MatrixProduct::GetSelfType(Sq0, q);

	auto R = TMD::ScalarMatrixProduct::GetSelfType(
		TMD::MatrixScalarPower::GetSelfType(
			TMD::GetDotProduct(q, q),
			TMD::RationalScalarConstant::GetSelfType(-1)
		),
		TMD::GetMultipleAddition({
			TMD::MatrixProduct::GetSelfType(
				v, TMD::Transpose::GetSelfType(v)
			),
			TMD::ScalarMatrixProduct::GetSelfType(
				TMD::MatrixProduct::GetSelfType(
					q0, TMD::Transpose::GetSelfType(q0)
				),
				TMD::IdentityMatrix::GetSelfType(3)
			),
			TMD::ScalarMatrixProduct::GetSelfType(
				TMD::ScalarMatrixProduct::GetSelfType(
					TMD::RationalScalarConstant::GetSelfType(2), q0
				),
				skew_v
			),
			TMD::MatrixProduct::GetSelfType(
				skew_v, skew_v
			)
		})
	);

	// P related
	auto x_of_P = TMD::Variable::GetSelfType("X", variable_id++, 3, 1);
	auto x1 = TMD::MatrixProduct::GetSelfType(Sx1, x_of_P);
	auto x2 = TMD::MatrixProduct::GetSelfType(Sx2, x_of_P);
	auto x3 = TMD::MatrixProduct::GetSelfType(Sx2, x_of_P);
	auto x3_inv = TMD::MatrixScalarPower::GetSelfType(x3, TMD::RationalScalarConstant::GetSelfType(-1));

	auto P = TMD::GetMultipleAddition({
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::ScalarMatrixProduct::GetSelfType(
				x1, x3_inv
			),
			TMD::ElementMatrix::GetSelfType(0, 0, 3, 1)
		),
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::ScalarMatrixProduct::GetSelfType(
				x2, x3_inv
			),
			TMD::ElementMatrix::GetSelfType(1, 0, 3, 1)
		),
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::GetVector2Norm(x_of_P),
			TMD::ElementMatrix::GetSelfType(2, 0, 3, 1)
		)
	});
	auto J = TMD::GetDerivative(P, x_of_P->_uuid);

	// gaussian related
	auto x_of_gaussian = TMD::Variable::GetSelfType("x", variable_id++, 2, 1);
	auto p_of_gaussian = TMD::Variable::GetSelfType("p", variable_id++, 2, 1);
	auto V_of_gaussian = TMD::Variable::GetSelfType("V", variable_id++, 2, 2);

	auto x_minus_p = TMD::MatrixAddition::GetSelfType(x_of_gaussian, TMD::Negate::GetSelfType(p_of_gaussian));

	auto exp_part = TMD::Exp::GetSelfType(
		TMD::ScalarMatrixProduct::GetSelfType(
			TMD::RationalScalarConstant::GetSelfType(TMD::RationalScalarConstant::RationalNumber(-1, 2)),
			TMD::GetMultipleProduct({
				TMD::Transpose::GetSelfType(x_minus_p),
				TMD::Inverse::GetSelfType(V_of_gaussian),
				x_minus_p
			})
		)
	);

	auto mult_part = TMD::MatrixScalarPower::GetSelfType(
		TMD::Determinant::GetSelfType(V_of_gaussian),
		TMD::RationalScalarConstant::GetSelfType(
			TMD::RationalScalarConstant::RationalNumber(-1, 2)
		)
	);

	auto gaussian = TMD::MatrixProduct::GetSelfType(mult_part, exp_part);

	// finally!
	auto Ap_plus_b = TMD::MatrixAddition::GetSelfType(
		TMD::MatrixProduct::GetSelfType(A, p), b
	);
	auto Jc = J->Substitute({{x_of_P->_uuid, Ap_plus_b}});
	auto SJARS = TMD::GetMultipleProduct({S2d, Jc, A, R, S});
	auto Vi = TMD::MatrixProduct::GetSelfType(SJARS, TMD::Transpose::GetSelfType(SJARS));
	auto pi = TMD::MatrixProduct::GetSelfType(S2d, P->Substitute({{x_of_P->_uuid, Ap_plus_b}}));
	
	// variables
	auto x = TMD::Variable::GetSelfType("x", variable_id++, 2, 1);
	auto alpha = gaussian->Substitute({
		{x_of_gaussian->_uuid, x},
		{p_of_gaussian->_uuid, pi},
		{V_of_gaussian->_uuid, Vi}
	});

	auto alpha_graph_str = alpha->ExportGraph();
	std::ofstream alpha_graph_file(fs::path(TEST_OUTPUT_DIR) / "alpha.tex");
	alpha_graph_file << alpha_graph_str;
	alpha_graph_file.close();

	auto alpha_gradient = TMD::GetDerivative(alpha, q->_uuid);
	auto alpha_gradient_graph_str = alpha_gradient->ExportGraph();
	std::ofstream alpha_gradient_file(fs::path(TEST_OUTPUT_DIR) / "alpha-gradient.tex");
	alpha_gradient_file << alpha_gradient_graph_str;
	alpha_gradient_file.close();

	TMD::VariableTable table;
	table[A->_uuid] = Eigen::MatrixXd::Random(3, 3);
	table[b->_uuid] = Eigen::MatrixXd::Random(3, 1);
	table[q->_uuid] = Eigen::MatrixXd::Random(4, 1);
	table[S->_uuid] = Eigen::MatrixXd::Random(3, 3);
	table[p->_uuid] = Eigen::MatrixXd::Random(3, 1);
	table[Sv->_uuid] = Eigen::MatrixXd::Random(3, 4);
	table[Sq0->_uuid] = Eigen::MatrixXd::Random(1, 4);
	table[Sx1->_uuid] = Eigen::MatrixXd::Random(1, 3);
	table[Sx2->_uuid] = Eigen::MatrixXd::Random(1, 3);
	table[Sx3->_uuid] = Eigen::MatrixXd::Random(1, 3);
	table[S2d->_uuid] = Eigen::MatrixXd::Random(2, 3);
	table[x->_uuid] = Eigen::MatrixXd::Random(2, 1);

	auto numeric_gradient = TMD::GetExpressionNumericDerivative(alpha, table, q->_uuid);
	auto analytic_gradient = alpha_gradient->SlowEvaluation(table);
	std::cerr << numeric_gradient.transpose() << std::endl
			  << analytic_gradient.transpose() << std::endl;

}