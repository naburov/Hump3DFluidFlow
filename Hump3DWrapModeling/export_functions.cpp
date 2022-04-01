#include <iostream>
#include <string>

#include <iostream>
#include <fstream>

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkVertex.h>
#include <vtkDoubleArray.h>
#include <vtkDataSet.h>
#include <vtkDataArray.h>
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>

#include "Consts.h"
#include "calculating_functions.h"
#include "export_functions.h"

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