#include <fstream>
#include <filesystem>
#include <iostream>
#include "Expressions.hpp"
#include "ExampleConfig.hpp"
#include "FiniteDifference.hpp"
#include <random>

namespace fs = std::filesystem;

// <- https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};


void WriteToFile(const fs::path& path, const std::string& str) {
	std::ofstream out(path);
	out << str;
	out.close();
}

double Gaussian(
	int dim,
	const Eigen::Ref<const Eigen::VectorXd>& x,
	const Eigen::Ref<const Eigen::VectorXd>& mean,
	const Eigen::Ref<const Eigen::MatrixXd>& variance
) {
	Eigen::VectorXd diff = x - mean;
	return std::exp(-0.5 * diff.transpose() * variance.inverse() * diff)
		   / std::sqrt(variance.determinant() * std::pow(2 * M_PI, dim));
}

int main(int argc, const char* argv[]) {
	const int dim = atoi(argv[1]);

	// definitions
	int variable_id = 0;
	auto variance = TMD::GetVariable("\\Sigma", variable_id++, dim, dim);
	auto mean = TMD::GetVariable("\\mu", variable_id++, dim, 1);
	auto x = TMD::GetVariable("x", variable_id++, dim, 1);
	auto diff = TMD::GetVariable("d", variable_id++, dim, 1);

	std::stringstream gaussian_constant_sstr;
	gaussian_constant_sstr << "(2\\pi)^{";
	if (dim & 1) {
		// odd
		gaussian_constant_sstr << "-" << dim << "/2}";
	} else {
		// even
		gaussian_constant_sstr << "-" << dim / 2 << "}";
	}
	auto gaussian_constant = TMD::GetVariable(gaussian_constant_sstr.str(), variable_id++, 1, 1);

	// expression
	auto gaussian_org = TMD::GetProduct({
		TMD::GetProduct({
			gaussian_constant,
			TMD::GetPower(
				TMD::GetDeterminant(variance),
				TMD::GetRationalScalarConstant(-1, 2)
			)
		}),
		TMD::GetExp(
			TMD::GetScalarProduct(
				TMD::GetRationalScalarConstant(-1, 2),
				TMD::GetProduct({
					TMD::GetTranspose(diff),
					TMD::GetInverse(variance),
					diff
				})
			)
		)
	});

	auto diff_expr = TMD::GetMinus(x, mean);
	auto gaussian = gaussian_org->Substitute({
		{diff->_variable_id, diff_expr}
	});

	WriteToFile(fs::path(EXAMPLE_OUTPUT_DIR) / "gaussian.tex", gaussian->ExportGraph());

	// derivatives
	auto gaussian_gradient = TMD::GetDerivative(gaussian, variance->_variable_id)->Simplify();
	WriteToFile(fs::path(EXAMPLE_OUTPUT_DIR) / "gaussian_gradient.tex", gaussian_gradient->ExportGraph());

	// verification
	TMD::VariableTable table;
	Eigen::MatrixXd sqrt_variance = variance->GetRandomMatrix();
	Eigen::MatrixXd variance_val = sqrt_variance.transpose() * sqrt_variance;
	Eigen::VectorXd mean_val = mean->GetRandomMatrix();
	normal_random_variable nrv(mean_val, variance_val);

	table[x->_variable_id] = nrv();
	table[mean->_variable_id] = mean_val;
	table[variance->_variable_id] = variance_val;
	table[gaussian_constant->_variable_id] = Eigen::Matrix<double, 1, 1>::Constant(std::pow(2 * M_PI, -(double)dim / 2));

	double numeric_gaussian = Gaussian(
		dim,
		table[x->_variable_id],
		table[mean->_variable_id],
		table[variance->_variable_id]
	);
	double analytic_gaussian = gaussian->SlowEvaluation(table)(0, 0);

	std::cerr << numeric_gaussian << " " << analytic_gaussian << std::endl;

	auto func = [&dim, &table, &x, &mean](const Eigen::MatrixXd& cur_variance) -> double {
		return Gaussian(
			dim,
			table[x->_variable_id],
			table[mean->_variable_id],
			cur_variance
		);
	};
	
	Eigen::VectorXd numeric_gradient = Numerics::ScalarGradient<double, Eigen::MatrixXd>(func, table[variance->_variable_id], 1e-4);
	Eigen::VectorXd analytic_gradient = gaussian_gradient->SlowEvaluation(table);

	std::cerr << numeric_gradient.transpose() << std::endl
			  << analytic_gradient.transpose() << std::endl;
}