#include "MatrixManipulation.hpp"
#define IMPLEMENT_SLOW_EVALUATION
#include "Expressions.hpp"
#include "FiniteDifference.hpp"
#include "ExampleConfig.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
const double PI = std::acos(-1);

double Gaussian2D(
	const Eigen::Matrix2d& V,
	const Eigen::Vector2d& x,
	const Eigen::Vector2d& p
) {
	double detV = V.determinant();
	Eigen::Vector2d x_minus_p = x - p;
	return 1.0 / (2 * PI) / std::sqrt(detV) * std::exp(-0.5 * x_minus_p.dot(V.inverse() * x_minus_p));
}

Eigen::Vector3d PerspectiveProject(
	const Eigen::Vector3d& x
) {
	return (Eigen::Vector3d() << x(0) / x(2), x(1) / x(2), x.norm()).finished();
}

Eigen::Vector2d GetPi(
	const Eigen::Matrix3d& A,
	const Eigen::Vector3d& b,
	const Eigen::Vector3d& p
) {
	Eigen::Vector3d pc = PerspectiveProject(A * p + b);
	Eigen::Vector2d pi = pc.head<2>();
	return pi;
}

Eigen::Matrix2d GetVi(
	const Eigen::Matrix3d& J,
	const Eigen::Matrix3d& A,
	const Eigen::Matrix3d& R,
	const Eigen::Matrix3d& S
) {
	Eigen::Matrix3d JARS = J * A * R * S;
	Eigen::Matrix3d V = JARS * JARS.transpose();
	Eigen::Matrix2d Vi = V.block<2, 2>(0, 0);
	return Vi;
}

Eigen::Matrix3d GetR(
	const Eigen::Vector4d& q
) {
	double q0 = q(0);
	Eigen::Vector3d v = q.tail<3>();
	Eigen::Matrix3d hat_v = Numerics::GetHatMatrix(v);
	Eigen::Matrix3d R =
		1.0 / q.squaredNorm() * (
			v * v.transpose() + q0 * q0 * Eigen::Matrix3d::Identity() + 2 * q0 * hat_v + hat_v * hat_v
		);
	return R;
}

double Alpha(
	const Eigen::Matrix3d& A,
	const Eigen::Vector3d& b,
	const Eigen::Vector2d& x,
	const Eigen::Vector4d& q,
	const Eigen::Matrix3d& S,
	const Eigen::Vector3d& p,
	const Eigen::Matrix3d& J
) {
	Eigen::Matrix3d R = GetR(q);
	Eigen::Matrix2d Vi = GetVi(J, A, R, S);
	Eigen::Vector2d pi = GetPi(A, b, p);

	return Gaussian2D(Vi, x, pi);
}

int main() {
	int variable_id = 0;
	
	// camera transformation
	auto A = TMD::GetVariable("A", variable_id++, 3, 3);
	auto b = TMD::GetVariable("b", variable_id++, 3, 1);

	// Gaussian configuration
	auto q = TMD::GetVariable("q", variable_id++, 4, 1);
	auto S = TMD::GetVariable("S", variable_id++, 3, 3);
	auto p = TMD::GetVariable("p", variable_id++, 3, 1);

	// Other constant
	auto gaussian_normalizer = TMD::GetVariable("\\frac{1}{2 \\pi}", variable_id++, 1, 1);

	// Selection Matrices
	auto Sv = TMD::GetVariable("\\mathcal{S}_v", variable_id++, 3, 4);
	auto Sq0 = TMD::GetVariable("\\mathcal{S}_{q_0}", variable_id++, 1, 4);
	auto Sx1 = TMD::GetVariable("\\mathcal{S}_{x_1}", variable_id++, 1, 3);
	auto Sx2 = TMD::GetVariable("\\mathcal{S}_{x_2}", variable_id++, 1, 3);
	auto Sx3 = TMD::GetVariable("\\mathcal{S}_{x_3}", variable_id++, 1, 3);
	auto S2d = TMD::GetVariable("\\mathcal{S}_{2d}", variable_id++, 2, 3);

	// R related
	auto v = TMD::GetProduct({Sv, q});
	auto skew_v = TMD::GetSkew(v);
	auto q0 = TMD::GetProduct({Sq0, q});

	auto R = TMD::GetScalarProduct(
		TMD::GetPower(
			TMD::GetDotProduct(q, q),
			TMD::GetRationalScalarConstant(-1)
		),
		TMD::GetAddition({
			TMD::GetProduct({
				v, TMD::GetTranspose(v)
			}),
			TMD::GetScalarProduct(
				TMD::GetProduct({
					q0, TMD::GetTranspose(q0)
				}),
				TMD::GetIdeneityMatrix(3)
			),
			TMD::GetScalarProduct(
				TMD::GetScalarProduct(
					TMD::GetRationalScalarConstant(2), q0
				),
				skew_v
			),
			TMD::GetProduct({
				skew_v, skew_v
			})
		})
	);

	// P related
	auto x_of_P = TMD::GetVariable("X", variable_id++, 3, 1);
	auto x1 = TMD::GetProduct({Sx1, x_of_P});
	auto x2 = TMD::GetProduct({Sx2, x_of_P});
	auto x3 = TMD::GetProduct({Sx3, x_of_P});
	auto x3_inv = TMD::GetPower(x3, TMD::GetRationalScalarConstant(-1));

	auto P = TMD::GetAddition({
		TMD::GetScalarProduct(
			TMD::GetScalarProduct(
				x1, x3_inv
			),
			TMD::GetSelectMatrix(0, 1, 3)
		),
		TMD::GetScalarProduct(
			TMD::GetScalarProduct(
				x2, x3_inv
			),
			TMD::GetSelectMatrix(1, 2, 3)
		),
		TMD::GetScalarProduct(
			TMD::GetVector2Norm(x_of_P),
			TMD::GetSelectMatrix(2, 3, 3)
		)
	});
	auto J = TMD::GetTranspose(TMD::GetDerivative(P, x_of_P->_variable_id));

	// gaussian related
	auto x_of_gaussian = TMD::GetVariable("x", variable_id++, 2, 1);
	auto p_of_gaussian = TMD::GetVariable("p", variable_id++, 2, 1);
	auto V_of_gaussian = TMD::GetVariable("V", variable_id++, 2, 2);

	auto x_minus_p = TMD::GetMinus(x_of_gaussian, p_of_gaussian);

	auto exp_part = TMD::GetExp(
		TMD::GetScalarProduct(
			TMD::GetRationalScalarConstant(-1, 2),
			TMD::GetProduct({
				TMD::GetTranspose(x_minus_p),
				TMD::GetInverse(V_of_gaussian),
				x_minus_p
			})
		)
	);

	auto mult_part = TMD::GetProduct({
		gaussian_normalizer,
		TMD::GetPower(
			TMD::GetDeterminant(V_of_gaussian),
			TMD::GetRationalScalarConstant(-1, 2)
		)
	});

	auto gaussian = TMD::GetProduct({mult_part, exp_part});

	auto gaussian_graph_str = TMD::GetDerivative(gaussian, V_of_gaussian->_variable_id)->ExportGraph();
	std::ofstream gaussian_graph_file(fs::path(EXAMPLE_OUTPUT_DIR) / "gaussian.tex");
	gaussian_graph_file << gaussian_graph_str;
	gaussian_graph_file.close();

	// finally!
	auto Ap_plus_b = TMD::GetAddition({
		TMD::GetProduct({A, p}), b
	});
	auto Jc = J->Substitute({{x_of_P->_variable_id, Ap_plus_b}});
	auto SJARS = TMD::GetProduct({S2d, Jc, A, R, S});
	auto Vi = TMD::GetProduct({SJARS, TMD::GetTranspose(SJARS)});
	auto pi = TMD::GetProduct({S2d, P->Substitute({{x_of_P->_variable_id, Ap_plus_b}})});
	
	// variables
	auto x = TMD::GetVariable("x", variable_id++, 2, 1);
	auto alpha = gaussian->Substitute({
		{x_of_gaussian->_variable_id, x},
		{p_of_gaussian->_variable_id, pi},
		{V_of_gaussian->_variable_id, Vi}
	});

	auto alpha_graph_str = alpha->ExportGraph();
	std::ofstream alpha_graph_file(fs::path(EXAMPLE_OUTPUT_DIR) / "alpha.tex");
	alpha_graph_file << alpha_graph_str;
	alpha_graph_file.close();

	auto alpha_gradient = TMD::GetDerivative(alpha, q->_variable_id)->Simplify();
	auto alpha_gradient_graph_str = alpha_gradient->ExportGraph();
	std::ofstream alpha_gradient_file(fs::path(EXAMPLE_OUTPUT_DIR) / "alpha-gradient.tex");
	alpha_gradient_file << alpha_gradient_graph_str;
	alpha_gradient_file.close();

	TMD::VariableTable table;
	table[A->_variable_id] = Eigen::MatrixXd::Random(3, 3);
	table[b->_variable_id] = Eigen::MatrixXd::Random(3, 1);
	table[q->_variable_id] = Eigen::MatrixXd::Random(4, 1);
	table[S->_variable_id] = Eigen::MatrixXd::Random(3, 3);
	table[p->_variable_id] = Eigen::MatrixXd::Random(3, 1);
	table[gaussian_normalizer->_variable_id] = (Eigen::MatrixXd(1, 1) << 1.0 / (2 * PI)).finished();

	table[Sv->_variable_id] = Eigen::MatrixXd::Zero(3, 4);
	table[Sq0->_variable_id] = Eigen::MatrixXd::Zero(1, 4);
	table[Sx1->_variable_id] = Eigen::MatrixXd::Zero(1, 3);
	table[Sx2->_variable_id] = Eigen::MatrixXd::Zero(1, 3);
	table[Sx3->_variable_id] = Eigen::MatrixXd::Zero(1, 3);
	table[S2d->_variable_id] = Eigen::MatrixXd::Zero(2, 3);

	table[Sv->_variable_id].rightCols<3>() = Eigen::Matrix3d::Identity();
	table[Sq0->_variable_id](0, 0) = 1;
	table[Sx1->_variable_id](0, 0) = 1;
	table[Sx2->_variable_id](0, 1) = 1;
	table[Sx3->_variable_id](0, 2) = 1;
	table[S2d->_variable_id].leftCols<2>() = Eigen::Matrix2d::Identity();
	
	table[x->_variable_id] = Eigen::MatrixXd::Random(2, 1);

	Eigen::Matrix3d J_matrix = Jc->SlowEvaluation(table);

	double numeric_val = Alpha(
		table[A->_variable_id],
		table[b->_variable_id],
		table[x->_variable_id],
		table[q->_variable_id],
		table[S->_variable_id],
		table[p->_variable_id],
		J_matrix
	);
	double analytic_val = alpha->SlowEvaluation(table)(0, 0);
	std::cerr << numeric_val << " " << analytic_val << std::endl;

	auto numeric_gradient = TMD::GetExpressionNumericDerivative(alpha, table, q->_variable_id);
	auto analytic_gradient = alpha_gradient->SlowEvaluation(table);
	std::cerr << numeric_gradient.transpose() << std::endl
			  << analytic_gradient.transpose() << std::endl;

	auto alpha_hessian = TMD::GetDerivative(alpha_gradient, q->_variable_id)->Simplify();
	auto alpha_hessian_graph_str = alpha_hessian->ExportGraph();
	std::ofstream alpha_hessian_file(fs::path(EXAMPLE_OUTPUT_DIR) / "alpha-hessian.tex");
	alpha_hessian_file << alpha_hessian_graph_str;
	alpha_hessian_file.close();

	auto numeric_hessian = TMD::GetExpressionNumericDerivative(alpha_gradient, table, q->_variable_id);
	auto analytic_hessian = alpha_hessian->SlowEvaluation(table);
	std::cerr << numeric_hessian << std::endl
			  << analytic_hessian << std::endl;

}