#include "Expressions.hpp"
#include <iostream>
#include "gtest/gtest.h"

TEST(DerivativeTest, ProductTest) {
	auto A = TMD::Variable::GetVariable("A", TMD::UUIDGenerator::GenUUID(), 3, 4);
	auto B = TMD::Variable::GetVariable("B", TMD::UUIDGenerator::GenUUID(), 5, 6);
	auto AxB = TMD::KroneckerProduct::GetKroneckerProduct(A, B);
	auto derivativeA = TMD::GetDerivative(AxB, A->_uuid);
	auto derivativeB = TMD::GetDerivative(AxB, B->_uuid);
	std::cerr << "Result for A: ";
	derivativeA->Print(std::cerr);
	std::cerr << std::endl;

	std::cerr << "Result for B: ";
	derivativeB->Print(std::cerr);
	std::cerr << std::endl;

	delete AxB;
	delete derivativeA;
	delete derivativeB;
}

TEST(DerivativeTest, KroneckerProductTest) {
	auto A = TMD::Variable::GetVariable("A", TMD::UUIDGenerator::GenUUID(), 3, 4);
	auto B = TMD::Variable::GetVariable("B", TMD::UUIDGenerator::GenUUID(), 5, 6);
	auto AKronB = TMD::KroneckerProduct::GetKroneckerProduct(A, B);
	auto derivativeA = TMD::GetDerivative(AKronB, A->_uuid);
	auto derivativeB = TMD::GetDerivative(AKronB, B->_uuid);
	std::cerr << "Result for A: ";
	derivativeA->Print(std::cerr);
	std::cerr << std::endl;

	std::cerr << "Result for B: ";
	derivativeB->Print(std::cerr);
	std::cerr << std::endl;

	delete AKronB;
	delete derivativeA;
	delete derivativeB;
}

TEST(DerivativeTest, AdditionTest) {
	// TODO:
}