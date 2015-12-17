#pragma once

struct active
{ float active_c0;
  vw* all;//statistics, loss
  bool oracular;
  bool simple_threshold;
  size_t max_labels;	
  size_t min_labels;	
};

float query_decision(active& a, example& ec, float k);
LEARNER::base_learner* active_setup(vw& all);
