#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <vector>
using std::vector;

using namespace std;
using namespace cv;

// for pixel
struct StructEllipsoid
{
	// recent data
	int			n;

	// this use to calculate a, b
	float		mu[3];

	// axis
	float		a;
	float		b;
	float		c;

};

class SBE
{
public:
	SBE(int img_rows, int img_cols);
	~SBE();

	// running method
	Mat		Run(Mat &image);

private:
	int imgNums;
	int imgRols, imgCols;

	// core data structure
	vector< vector <StructEllipsoid> >	codeEllipsoid;
	// other data structure
	vector< vector <Vec3d> >			codeData;		
	float								globalMinR;

	// for pixel
	inline float DetectPixel(int &k, vector<StructEllipsoid> &code_ellipsoid, Vec3d code_data);	// return result image
	inline bool CreateEllipsoid(vector<Vec3d> &code_data, vector<StructEllipsoid> &code_ellipsoid);
	inline bool UpdateNeighbor(vector<StructEllipsoid> &code_ellipsoid, Vec3d code_mu, float minr);
	inline bool DeleteEllipsoid(vector<StructEllipsoid> &code_ellipsoid, int bg_nums);
	inline bool CheckPixelData_Variance(vector<Vec3d> &input_data, vector<Vec3d> &output_data);
	inline bool CheckPixelData_Medium(vector<Vec3d> &input_data, vector<Vec3d> &output_data);

	// for each ellipsoid
	inline bool CalcEllipsoid(StructEllipsoid &el, float* sigmax, float* sigmax2);
	inline bool UpdateEllipsoid(Vec3d data_now, StructEllipsoid &el);
	inline bool CreateSphere(StructEllipsoid &sel, vector<Vec3d> sdata);
	inline bool AddEllipsoid(vector<StructEllipsoid> &sel, StructEllipsoid &el);

	// tools
	int getRandomNum(int ssize);
	int GetNeighborPos(int i, int j, int rows, int cols);
	int GetCloseMean(int k, Vec3d temp_data, vector<Vec3d> &mu);
	void GetNeighborPosOffset(int r, int n, vector<Vec2f> &npointoffset);
	bool DetectCombine(vector<Vec3d> &vec_pixel, Vec3d now_pixel);
	float GetMinAxis(StructEllipsoid &el);
	void GetMedium(vector<Vec3d> &sdata, double *medium);

};
