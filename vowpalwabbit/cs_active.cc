#include <errno.h>
#include "reductions.h"
#include "v_hashmap.h"
#include "rand48.h"
#include "float.h"
#include "vw.h"
#include "vw_exception.h"
#include "csoaa.h"
#define B_SEARCH_MAX_ITER 20

using namespace LEARNER;
using namespace std;
using namespace COST_SENSITIVE;

struct cs_active
{ // active learning algorithm parameters
  float c0; // mellowness controlling the width of the set of good functions
  float c1; // multiplier on the threshold for the cost range test
  float cost_max; // max cost
  float cost_min; // min cost

  uint32_t num_classes;
  size_t t;

  size_t min_labels;
  size_t max_labels;

  bool print_debug_stuff;
  bool is_baseline;  

  vw* all;//statistics, loss
  LEARNER::base_learner* l;
};

float binarySearch(float fhat, float delta, float sens, float tol)
{ float maxw = min(fhat/sens,FLT_MAX);
  
  if(maxw*fhat*fhat <= delta)
  { return maxw;
  }

  float l = 0, u = maxw, w, v;
  
  for(int iter=0; iter<B_SEARCH_MAX_ITER; iter++)
  { w = (u+l)/2.f;
    v = w*(fhat*fhat-(fhat-sens*w)*(fhat-sens*w))-delta;
    if(v > 0)
    { u = w;
    }
    else
    { l = w;
    }
    if(fabs(v)<=tol || u-l<=tol)
    { break;
    }
  }

  return l;
}

template<bool is_learn, bool is_simulation>
inline void inner_loop(cs_active& cs_a, base_learner& base, example& ec, uint32_t i, float cost,
                       uint32_t& prediction, float& score, float& partial_prediction, bool query_this_label, bool& query_needed)
{ base.predict(ec, i-1);
  //cerr << "base.predict ==> partial_prediction=" << ec.partial_prediction << endl;
  if (is_learn)
  { vw& all = *cs_a.all;
    ec.l.simple.weight = 1.;
    ec.weight = 1.;
    if (is_simulation)
    { // In simulation mode 
      if(query_this_label)
      { ec.l.simple.label = cost;
        all.sd->queries += 1;
      }
      else 
      { ec.l.simple.label = FLT_MAX;
      }
    }
    else
    { // In reduction mode.
      // If the cost of this label was previously queried, then it should be available for learning now.
      // If the cost of this label was not queried, then skip it. 
      if (query_needed)
      { ec.l.simple.label = cost;
        if ((cost < cs_a.cost_min) || (cost > cs_a.cost_max))
          cerr << "warning: cost " << cost << " outside of cost range [" << cs_a.cost_min << ", " << cs_a.cost_max << "]!" << endl;
      }
      else
      { ec.l.simple.label = FLT_MAX;
      }
    }

    if(ec.l.simple.label != FLT_MAX)
    {
      //cerr << "t = " << cs_a.t << ", base.learn(" << i-1 << ", " << ec.l.simple.label << ", " << ec.l.simple.weight  << ")" << endl;
      base.learn(ec, i-1);
    } //else
    //cerr << "t = " << cs_a.t << ", no base.learn(" << i-1 << ", " << ec.l.simple.label << ")" << endl;
  }
  else if (!is_simulation) 
  {// Prediction in reduction mode could be used by upper layer to ask whether this label needs to be queried.
   // So we return that.
    query_needed = query_this_label;
  }
  
  partial_prediction = ec.partial_prediction;
  if (ec.partial_prediction < score || (ec.partial_prediction == score && i < prediction))
  { score = ec.partial_prediction;
    prediction = i;
  }
  add_passthrough_feature(ec, i, ec.partial_prediction);
}

inline void find_cost_range(cs_active& cs_a, base_learner& base, example& ec, uint32_t i, float delta, float eta, float& min_pred, float& max_pred, bool& is_range_large)
{ float tol = 1e-6;

  base.predict(ec, i-1);
  float sens = base.sensitivity(ec, i-1);
  
  if (cs_a.t <= 1 || nanpattern(sens) || infpattern(sens))
  { min_pred = cs_a.cost_min;
    max_pred = cs_a.cost_max;
    is_range_large = true;
    cerr << "  find_cost_rangeA: i=" << i << " pp=" << ec.partial_prediction << " sens=" << sens << " eta=" << eta << " [" << min_pred << ", " << max_pred << "] = " << (max_pred-min_pred) << endl;
  }
  else
  { // finding max_pred and min_pred by binary search
    max_pred =  min(ec.pred.scalar + sens*binarySearch(cs_a.cost_max-ec.pred.scalar, delta, sens, tol), cs_a.cost_max);
    min_pred =  max(ec.pred.scalar - sens*binarySearch(ec.pred.scalar-cs_a.cost_min, delta, sens, tol), cs_a.cost_min);
    is_range_large = (max_pred-min_pred > eta);
    cerr << "  find_cost_rangeB: i=" << i << " pp=" << ec.partial_prediction << " sens=" << sens << " eta=" << eta << " [" << min_pred << ", " << max_pred << "] = " << (max_pred-min_pred) << endl;
  }
}

template <bool is_learn, bool is_simulation>
void predict_or_learn(cs_active& cs_a, base_learner& base, example& ec)
{ //cerr << "------------- passthrough" << endl;
  COST_SENSITIVE::label ld = ec.l.cs;

  //cerr << "is_learn=" << is_learn << " ld.costs.size()=" << ld.costs.size() << endl;
  if(cs_a.all->sd->queries >= cs_a.min_labels*cs_a.num_classes)
  { // save regressor
    stringstream filename;
    filename << cs_a.all->final_regressor_name << "." << ec.example_t << "." << cs_a.all->sd->queries;	
    VW::save_predictor(*(cs_a.all), filename.str());
  
    // Double label query budget	
    cs_a.min_labels *= 2;
    for (size_t i=0; i<cs_a.all->sd->examples_by_queries.size(); i++)
    {  cerr << endl << "examples with " << i << " labels queried = " << cs_a.all->sd->examples_by_queries[i];
    }

    cerr << endl << "labels outside of cost range = " << cs_a.all->sd->labels_outside_range;
    cerr << endl << "average distance to range = " << cs_a.all->sd->distance_to_range/((float)cs_a.all->sd->labels_outside_range);
    cerr << endl << "average range = " << cs_a.all->sd->range/((float)cs_a.all->sd->labels_outside_range);
 
    /*
    for (size_t i=0; i<cs_a.all->sd->distance_to_range.size(); i++)
    {  cerr << endl << "label " << i << ", average distance to range = " << cs_a.all->sd->distance_to_range[i]/((float)(cs_a.t-1));
    }*/

    cerr << endl << endl;

  }

  if(cs_a.all->sd->queries >= cs_a.max_labels*cs_a.num_classes)
  { return;
  }

  uint32_t prediction = 1;
  float score = FLT_MAX;
  ec.l.simple = { 0., 0., 0. };

  float min_max_cost = FLT_MAX;
  float t = (float)cs_a.t; // ec.example_t;  // current round
  float t_prev = t-1.; // ec.weight; // last round

  float eta = cs_a.c1*(cs_a.cost_max - cs_a.cost_min)/sqrt(t); // threshold on cost range
  float delta = cs_a.c0*log((float)(cs_a.num_classes*max(t_prev,1.f)))*pow(cs_a.cost_max-cs_a.cost_min,2);  // threshold on empirical loss difference

  if (ld.costs.size() > 0)
  { uint32_t n_overlapped = 0;
    { for (COST_SENSITIVE::wclass& cl : ld.costs)
      { find_cost_range(cs_a, base, ec, cl.class_index, delta, eta, cl.min_pred, cl.max_pred, cl.is_range_large);
        min_max_cost = min(min_max_cost, cl.max_pred);
      }
    }

    for (COST_SENSITIVE::wclass& cl : ld.costs)
    { cl.is_range_overlapped = (cl.min_pred <= min_max_cost);
      n_overlapped += (uint32_t)(cl.is_range_overlapped); 
        //large_range = large_range || (cl.is_range_overlapped && cl.is_range_large);
        //if(cl.is_range_overlapped && is_learn)
        //{ cout << "label " << cl.class_index << ", min_pred = " << cl.min_pred << ", max_pred = " << cl.max_pred << ", is_range_large = " << cl.is_range_large << ", eta = " << eta << ", min_max_cost = " << min_max_cost << endl;
        //}
      cs_a.all->sd->overlapped_and_range_small += (size_t)(cl.is_range_overlapped && !cl.is_range_large);
      if(cl.x > cl.max_pred || cl.x < cl.min_pred)
      {  cs_a.all->sd->labels_outside_range++;
         //cs_a.all->sd->distance_to_range[cl.class_index-1] += max(cl.x - cl.max_pred, cl.min_pred - cl.x);
         cs_a.all->sd->distance_to_range += max(cl.x - cl.max_pred, cl.min_pred - cl.x);
         cs_a.all->sd->range += cl.max_pred - cl.min_pred;
      }

    }
    
    bool query = (n_overlapped > 1);
    size_t queries = cs_a.all->sd->queries;
    for (COST_SENSITIVE::wclass& cl : ld.costs)
    { bool query_label = (query && (cs_a.is_baseline || (cl.is_range_overlapped && cl.is_range_large)));
      inner_loop<is_learn,is_simulation>(cs_a, base, ec, cl.class_index, cl.x, prediction, score, cl.partial_prediction, query_label, cl.query_needed);
      if (cs_a.print_debug_stuff)
        cerr << "label=" << cl.class_index << " x=" << cl.x << " prediction=" << prediction << " score=" << score << " pp=" << cl.partial_prediction << " ql=" << query_label << " qn=" << cl.query_needed << " ro=" << cl.is_range_overlapped << " rl=" << cl.is_range_large << " [" << cl.min_pred << ", " << cl.max_pred << "] vs delta=" << delta << " n_overlapped=" << n_overlapped << " is_baseline=" << cs_a.is_baseline << endl;
    }

    cs_a.all->sd->examples_by_queries[cs_a.all->sd->queries - queries] += 1;

    ec.partial_prediction = score;
    if(is_learn)
    { cs_a.t++;
    }
  }
  else
  { float temp = 0.f;
    bool temp2=false, temp3=false;
    for (uint32_t i = 1; i <= cs_a.num_classes; i++)
    { inner_loop<false,is_simulation>(cs_a, base, ec, i, FLT_MAX, prediction, score, temp, temp2, temp3);
    }
  }

  ec.pred.multiclass = prediction;
  ec.l.cs = ld;
}

void finish_example(vw& all, cs_active& cs_a, example& ec)
{ CSOAA::finish_example(all, *(CSOAA::csoaa*)&cs_a, ec);
}

base_learner* cs_active_setup(vw& all)
{ //parse and set arguments
  if(missing_option<size_t, true>(all, "cs_active", "Cost-sensitive active learning with <k> costs"))
    return nullptr;

  new_options(all, "cost-sensitive active Learning options")
  ("simulation", "cost-sensitive active learning simulation mode")
  ("baseline", "cost-sensitive active learning baseline")
  ("mellowness",po::value<float>(),"mellowness parameter c_0. Default 0.1.")
  ("range_c", po::value<float>(),"parameter controlling the threshold for per-label cost uncertainty. Default 0.5.")
  ("max_labels", po::value<float>(), "maximum number of label queries.")
  ("min_labels", po::value<float>(), "minimum number of label queries.")
  ("cost_max",po::value<float>(),"cost upper bound. Default 1.")
  ("cost_min",po::value<float>(),"cost lower bound. Default 0.")
  ("csa_debug", "print debug stuff for cs_active");
    ;
  add_options(all);

  cs_active& data = calloc_or_throw<cs_active>();
  
  data.num_classes = (uint32_t)all.vm["cs_active"].as<size_t>();
  data.c0 = 0.1;
  data.c1 = 0.5;
  data.all = &all;
  data.cost_max = 1.f;
  data.cost_min = 0.f;
  data.t = 1;
  data.max_labels = (size_t)-1;
  data.min_labels = (size_t)-1;
  data.is_baseline = false; 
  data.print_debug_stuff = all.vm.count("csa_debug") > 0;
  
  if(all.vm.count("baseline"))
  { data.is_baseline = true;
  }
  
  if(all.vm.count("mellowness"))
  { data.c0 = all.vm["mellowness"].as<float>();
  }

  if(all.vm.count("range_c"))
  { data.c1 = all.vm["range_c"].as<float>();
  }
  
  if(all.vm.count("cost_max"))
  { data.cost_max = all.vm["cost_max"].as<float>();
  }
  
  if(all.vm.count("cost_min"))
  { data.cost_min = all.vm["cost_min"].as<float>();
  }

  string loss_function = all.vm["loss_function"].as<string>();
  if (loss_function.compare("squared") != 0)
  { free(&data);
    THROW("error: you can't use non-squared loss with cs_active");
  }
  
  if (count(all.args.begin(), all.args.end(),"--lda") != 0)
  { free(&data);
    THROW("error: you can't combine lda and active learning");
  }
 
  if(all.vm.count("max_labels"))
  { data.max_labels = (size_t)all.vm["max_labels"].as<float>();
  }
  
  if(all.vm.count("min_labels"))
  { data.min_labels = (size_t)all.vm["min_labels"].as<float>();
  }
  
  if (count(all.args.begin(), all.args.end(),"--active") != 0)
  { free(&data);
    THROW("error: you can't use --cs_active and --active at the same time");
  }
  
  if (count(all.args.begin(), all.args.end(),"--active_cover") != 0)
  { free(&data);
    THROW("error: you can't use --cs_active and --active_cover at the same time");
  }
  
  if (count(all.args.begin(), all.args.end(),"--csoaa") != 0)
  { free(&data);
    THROW("error: you can't use --cs_active and --csoaa at the same time");
  }

  learner<cs_active>* l; 

  if (all.vm.count("simulation"))
    l = &init_learner(&data, setup_base(all), predict_or_learn<true,true>, predict_or_learn<false,true>, data.num_classes);
  else
    l = &init_learner(&data, setup_base(all), predict_or_learn<true,false>, predict_or_learn<false,false>, data.num_classes);

  all.set_minmax(all.sd,data.cost_max);
  all.set_minmax(all.sd,data.cost_min);
  //cerr << "cs_active data = " << & data << endl;

  all.p->lp = cs_label; // assigning the label parser
  for (uint32_t i=0; i<data.num_classes+1; i++)
    all.sd->examples_by_queries.push_back(0); 

  /*
  for (uint32_t i=0; i<data.num_classes; i++)
    all.sd->distance_to_range.push_back(0.f); 
  */

  l->set_finish_example(finish_example);
  base_learner* b = make_base(*l);
  all.cost_sensitive = b;
  return b;
}
