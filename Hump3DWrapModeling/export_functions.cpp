#include <iostream>
#include <string>

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkVertex.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>

#include "calculating_functions.h"
#include "export_functions.h"
#include "consts.h"

void print_array(double*** arr, const int(&dims)[3]) {
	for (int i = 0; i < dims[0]; ++i)
	{
		for (int j = 0; j < dims[1]; ++j)
		{
			for (int k = 0; k < dims[2]; ++k)
				std::cout << arr[i][j][k] << ' ';
		}
		std::cout << std::endl;
	}
}

void dispose_array(double*** arr, const int(&dims)[3]) {
	for (int i = 0; i < dims[0]; ++i)
	{
		for (int j = 0; j < dims[1]; ++j)
		{
			delete[] arr[i][j];
		}
		delete[] arr[i];
	}
}

void print_min_max_values(double*** arr, std::string name, const int(&dims)[3])
{
	std::vector<double> s;
	for (int i = 0; i < dims[0]; ++i)
	{
		for (int j = 0; j < dims[1]; ++j)
		{
			for (size_t k = 0; k < dims[2]; k++)
			{
				s.push_back(arr[i][j][k]);
			}
		}
	}
	std::cout << " " + name + " max " << *std::max_element(s.begin(), s.end()) << " " + name + " min " << *std::min_element(s.begin(), s.end()) << std::endl;
}

void export_grid(const std::string& filename, const double(&deltas)[3]) {
	vtkNew<vtkStructuredGrid> sgrid;
	int n = N * K;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

	int id = 0;
	for (int i = 0; i < N; ++i) {
		double xi1 = xi1_min + deltas[0] * i;
		for (size_t k = 0; k < K; k++) {
			double xi2 = xi2_min + deltas[2] * k;
			double theta = theta_min;
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2), xi2);
			id++;

		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(N, 1, K);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_central_slice(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3])
{
	vtkNew<vtkStructuredGrid> sgrid;
	int n = N * M * 1;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int k = K / 2 + 1;
	for (int i = 0; i < N; ++i) {
		double xi1 = xi1_min + deltas[0] * i;
		double xi2 = xi2_min + deltas[2] * k;
		for (size_t j = 0; j < M; j++) {
			double theta = theta_min + deltas[1] * j;
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2), xi2);
			u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
			id++;
		}

	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(N, M, 1);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_single_line(const std::string& filename, double*** T, const double(&deltas)[3], double theta)
{
	vtkNew<vtkStructuredGrid> sgrid;
	int n = N * 1 * 1;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int j = (theta - theta_min) / deltas[1];
	int k = K / 2 + 1;
	for (int i = 0; i < N; ++i) {
		double xi1 = xi1_min + deltas[0] * i;
		double xi2 = xi2_min + deltas[2] * k;
		double theta = theta_min + deltas[1] * j;
		points->InsertNextPoint(xi1, theta + mu(xi1, xi2), xi2);
		u->InsertNextTuple3(T[i][j][k], 0, 0);
		id++;
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(N, 1, 1);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_theta_slice(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3], double theta) {
	vtkNew<vtkStructuredGrid> sgrid;
	int n = N * 1 * K;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int j = (theta - theta_min) / deltas[1];
	for (int i = 0; i < N; ++i) {
		double xi1 = xi1_min + deltas[0] * i;
		for (size_t k = 0; k < K; k++) {
			double xi2 = xi2_min + deltas[2] * k;
			double theta = theta_min + deltas[1] * j;
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2), xi2);
			u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
			id++;

		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(N, 1, K);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3])
{
	vtkNew<vtkStructuredGrid> sgrid;
	int n = N * M * K;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	for (int i = 0; i < N; ++i) {
		double xi1 = xi1_min + deltas[0] * i;
		for (size_t k = 0; k < K; k++) {
			double xi2 = xi2_min + deltas[2] * k;
			for (size_t j = 0; j < M; j++) {
				double theta = theta_min + deltas[1] * j;
				points->InsertNextPoint(xi1, theta + mu(xi1, xi2), xi2);
				u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
				id++;
			}
		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(N, M, K);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

template <typename T>
void SwapEnd(T& var)
{
	char* varArray = reinterpret_cast<char*>(&var);
	for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
		std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

void output_vtk_binary_2d(const std::string& filename, double*** U, double*** V, int k, const double(&deltas)[3])
{

	int n1 = N;
	int n2 = M;

	float* my_u = new float[N * M];
	float* my_v = new float[N * M];

	for (int j = 0; j < n2; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			my_u[j * (n1) + i] = U[i][j][K / 2 + 1];
			SwapEnd(my_u[j * (n1) + i]);

			my_v[j * (n1) + i] = V[i][j][K / 2 + 1];
			SwapEnd(my_v[j * (n1) + i]);
		}
	}

	std::ofstream fileD;

	fileD.open(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	fileD << "# vtk DataFile Version 2.0" << "\n";
	fileD << "THIN_HUMP" << "\n";
	fileD << "BINARY" << "\n";
	fileD << "DATASET STRUCTURED_GRID" << std::endl;
	fileD << "DIMENSIONS " << n1 << " " << n2<< " " << "1" << std::endl;
	fileD << "POINTS " << N * M << " float" << std::endl;
	float tt1, tt2, tt3 = 0;
	SwapEnd(tt3);

	for (int j = 0; j < n2; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			tt1 = xi1_min + i * deltas[0];
			tt2 = theta_min + j * deltas[1] + mu(xi1_min + i * deltas[0], xi2_min + (K / 2 + 1) * deltas[2]);
			SwapEnd(tt1);
			SwapEnd(tt2);

			fileD.write((char*)&tt1, sizeof(float));
			fileD.write((char*)&tt2, sizeof(float));
			fileD.write((char*)&tt3, sizeof(float));

		}
	}
	fileD << "POINT_DATA " << N * M << std::endl;



	fileD << "VECTORS vvv float" << std::endl;

	for (int j = 0; j < n2; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			fileD.write((char*)&my_u[j * (n1) + i], sizeof(float));
			fileD.write((char*)&my_v[j * (n1) + i], sizeof(float));
			fileD.write((char*)&tt3, sizeof(float));

		}
	}
	fileD.close();


	delete[] my_u;
	delete[] my_v;

}