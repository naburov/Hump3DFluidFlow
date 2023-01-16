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

#include "cell_calculating_functions.h"
#include "export_functions.h"

void print_array(double*** arr, SimulationParams params) {
	for (int i = 0; i < params.dims[0]; ++i)
	{
		for (int j = 0; j < params.dims[1]; ++j)
		{
			for (int k = 0; k < params.dims[2]; ++k)
				std::cout << arr[i][j][k] << ' ';
		}
		std::cout << std::endl;
	}
}

void dispose_array(double*** arr, SimulationParams params) {
	for (int i = 0; i < params.dims[0]; ++i)
	{
		for (int j = 0; j < params.dims[1]; ++j)
		{
			delete[] arr[i][j];
		}
		delete[] arr[i];
	}
}

void print_min_max_values(double*** arr, std::string name, SimulationParams params)
{
	std::vector<double> s;
	for (int i = 0; i < params.dims[0]; ++i)
	{
		for (int j = 0; j < params.dims[1]; ++j)
		{
			for (size_t k = 0; k < params.dims[2]; k++)
			{
				s.push_back(arr[i][j][k]);
			}
		}
	}
	std::cout << " " + name + " max " << *std::max_element(s.begin(), s.end()) << " " + name + " min " << *std::min_element(s.begin(), s.end()) << std::endl;
}

void export_grid(const std::string& filename, SimulationParams params) {
	vtkNew<vtkStructuredGrid> sgrid;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

	int id = 0;
	for (int i = 0; i < params.dims[0]; ++i) {
		double xi1 = params.mins[0] + params.deltas[0] * i;
		for (size_t k = 0; k <params.dims[1]; k++) {
			double xi2 = params.mins[2] + params.deltas[2] * k;
			double theta = params.mins[1];
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2, params), xi2);
			id++;

		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(params.dims[0], 1,params.dims[1]);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_central_slice(const std::string& filename, double*** U, double*** V, double*** W, SimulationParams params)
{
	vtkNew<vtkStructuredGrid> sgrid;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int k =params.dims[1] / 2 + 1;
	for (int i = 0; i < params.dims[0]; ++i) {
		double xi1 = params.mins[0] + params.deltas[0] * i;
		double xi2 = params.mins[2] + params.deltas[2] * k;
		for (size_t j = 0; j <params.dims[2]; j++) {
			double theta = params.mins[1] + params.deltas[1] * j;
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2, params), xi2);
			u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
			id++;
		}

	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(params.dims[0],params.dims[2], 1);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_single_line(const std::string& filename, double*** T, double theta, SimulationParams params)
{
	vtkNew<vtkStructuredGrid> sgrid;
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int j = (theta - params.mins[1]) / params.deltas[1];
	int k = params.dims[2] / 2 + 1;
	for (int i = 0; i <params.dims[0]; ++i) {
		double xi1 = params.mins[0] + params.deltas[0] * i;
		double xi2 = params.mins[2] + params.deltas[2] * k;
		double theta = params.mins[1] + params.deltas[1] * j;
		points->InsertNextPoint(xi1, theta + mu(xi1, xi2, params), xi2);
		u->InsertNextTuple3(T[i][j][k], 0, 0);
		id++;
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(params.dims[0], 1, 1);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_theta_slice(const std::string& filename, double*** U, double*** V, double*** W, SimulationParams params, double theta) {
	vtkNew<vtkStructuredGrid> sgrid;
	int n =params.dims[0] * 1 *params.dims[1];
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	int j = (theta - params.mins[1]) / params.deltas[1];
	for (int i = 0; i <params.dims[0]; ++i) {
		double xi1 = params.mins[0] + params.deltas[0] * i;
		for (size_t k = 0; k <params.dims[1]; k++) {
			double xi2 = params.mins[2] + params.deltas[2] * k;
			double theta = params.mins[1] + params.deltas[1] * j;
			points->InsertNextPoint(xi1, theta + mu(xi1, xi2, params), xi2);
			u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
			id++;

		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(params.dims[0], 1,params.dims[1]);
	sgrid->GetPointData()->SetVectors(u);
	sgrid->SetPoints(points);

	vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
		vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(sgrid);
	writer->Write();
}

void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, SimulationParams params)
{
	vtkNew<vtkStructuredGrid> sgrid;
	int n =params.dims[0] *params.dims[2] *params.dims[1];
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> u = vtkSmartPointer<vtkDoubleArray>::New();
	u->SetName("u");
	u->SetNumberOfComponents(3);

	int id = 0;
	for (int i = 0; i <params.dims[0]; ++i) {
		double xi1 = params.mins[0] + params.deltas[0] * i;
		for (size_t k = 0; k <params.dims[1]; k++) {
			double xi2 = params.mins[2] + params.deltas[2] * k;
			for (size_t j = 0; j <params.dims[2]; j++) {
				double theta = params.mins[1] + params.deltas[1] * j;
				points->InsertNextPoint(xi1, theta + mu(xi1, xi2, params), xi2);
				u->InsertNextTuple3(U[i][j][k], V[i][j][k], W[i][j][k]);
				id++;
			}
		}
	}
	//std::cout << id << std::endl;
	sgrid->SetDimensions(params.dims[0],params.dims[2],params.dims[1]);
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

void output_vtk_binary_2d(const std::string& filename, double*** U, double*** V, int k, SimulationParams params)
{

	auto n1 = params.dims[0];
	auto n2 = params.dims[2];

	auto my_u = new float[params.dims[0] *params.dims[2]];
	auto my_v = new float[params.dims[0] *params.dims[2]];

	for (int j = 0; j < n2; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			my_u[j * (n1) + i] = U[i][j][params.dims[1] / 2 + 1];
			SwapEnd(my_u[j * (n1) + i]);

			my_v[j * (n1) + i] = V[i][j][params.dims[1] / 2 + 1];
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
	fileD << "POINTS " <<params.dims[0] *params.dims[2] << " float" << std::endl;
	float tt1, tt2, tt3 = 0;
	SwapEnd(tt3);

	for (int j = 0; j < n2; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			tt1 = params.mins[0] + i * params.deltas[0];
			tt2 = params.mins[1] + j * params.deltas[1] + mu(params.mins[0] + i * params.deltas[0], params.mins[2] + (params.dims[1] / 2 + 1) * params.deltas[2], params);
			SwapEnd(tt1);
			SwapEnd(tt2);

			fileD.write((char*)&tt1, sizeof(float));
			fileD.write((char*)&tt2, sizeof(float));
			fileD.write((char*)&tt3, sizeof(float));

		}
	}
	fileD << "POINT_DATA " <<params.dims[0] *params.dims[2] << std::endl;



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