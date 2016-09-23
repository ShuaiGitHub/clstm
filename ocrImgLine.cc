#include "clstm.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>
#include "clstmhl.h"
#include "extras.h"
#include "pstring.h"
#include "utils.h"

using namespace Eigen;
using namespace ocropus;
using std::vector;
using std::map;
using std::make_pair;
using std::shared_ptr;
using std::unique_ptr;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
#define string std_string
#define wstring std_wstring

inline float scaled_log(float x) {
  const float thresh = 10.0;
  if (x <= 0.0) return 0.0;
  float l = log(x);
  if (l < -thresh) return 0.0;
  if (l > 0) return 1.0;
  return (l + thresh) / thresh;
}

int main1(int argc, char **argv) {
  if (argc != 3) THROW("give text file as an argument; give outputfileName");

  string load_name = getsenv("load", "");
  if (load_name == "") THROW("must give load= parameter");
  CLSTMOCR clstm;
  clstm.load(load_name);

  bool conf = false;
  string output = "text";
  bool save_text = false;
  string line;
    Tensor2 raw;
    string fname(argv[1]);
    string outname(argv[2]);
    string basename = fname.substr(0, fname.find_last_of("."));
    read_png(raw, fname.c_str());
    raw() = -raw() + Float(1.0);
    string out = clstm.predict_utf8(raw());
    //ofs<<out<<endl;
    //ofs.close();
    write_text(outname, out);
    if (output == "text") {
      // nothing else to do
    } else {
      THROW("unknown output format");
    }
  return 0;
}

int main(int argc, char **argv) {
  TRY { return main1(argc, argv); }
  CATCH(const char *message) { cerr << "FATAL: " << message << endl; }
}
