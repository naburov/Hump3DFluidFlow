#include <iostream>
#include <string>

#include <iostream>
#include <fstream>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkArrowSource.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkProperty.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>

#include "Consts.h"
#include "calculating_functions.h"

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

void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3])
{
	double p[3], v[3];
	vtkNew<vtkImageData> image;
	image->SetDimensions(N, M, K);
	image->AllocateScalars(VTK_FLOAT, 3);

	for (size_t x = 0; x < N; x++)
	{
		for (size_t z = 0; z < K; z++)
		{
			for (size_t y = 0; y < M; y++)
			{
				auto pixel = static_cast<float*>(image->GetScalarPointer(x, y, z));
				pixel[0] = 0.0f;
				pixel[1] = 0.0f;
				pixel[2] = 0.0f;
			}
		}
	}

	for (size_t x = 0; x < N; x++)
	{
		for (size_t z = 0; z < K; z++)
		{
			double xi1 = x * deltas[0] + xi1_min;
			double xi2 = z * deltas[2] + xi2_min;
			for (size_t y = 0; theta_min + deltas[1] * y < theta_max - mu(xi1, xi2); y++)
			{
				auto y_shifted = y + static_cast<size_t>(mu(xi1, xi2) / deltas[1]);
				if (y_shifted > K - 1)
					continue;
				auto pixel = static_cast<float*>(image->GetScalarPointer(x, y_shifted, z));
				pixel[0] = (float)(U[x][y_shifted][z]);
				pixel[1] = (float)(V[x][y_shifted][z]);
				pixel[2] = (float)(W[x][y_shifted][z]);
			}
		}
	}

	image->GetPointData()->SetActiveVectors(
		image->GetPointData()->GetScalars()->GetName());

	vtkNew<vtkXMLImageDataWriter> writer;
	writer->SetFileName(filename.c_str());
	writer->SetInputData(image);
	writer->Write();
}