// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include "libvis/command_line_parser.h"

#include <iostream>
#include <sstream>

#include "libvis/logging.h"

namespace vis {

CommandLineParser::CommandLineParser(int argc, char** argv)
    : is_input_complete_(true),
      sequential_parameter_read_(false),
      argc_(argc),
      argv_(argv) {
  value_used_.resize(argc_, false);
  if (argc_ > 0) {
    value_used_[0] = true;
  }
  
  string help_param_1 = "-h";
  string help_param_2 = "--help";
  help_requested_ = false;
  for (int i = 1; i < argc; ++ i) {
    if (argv[i] == help_param_1 || argv[i] == help_param_2) {
      value_used_[i] = true;
      help_requested_ = true;
    }
  }
}

bool CommandLineParser::Flag(const char* name, const char* help) {
  CHECK(!sequential_parameter_read_) << "Flags must be read before all sequential parameters";
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = true;
  parameters_.back().is_sequential = false;
  parameters_.back().required = false;
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  string name_str = name;
  for (int i = 1; i < argc_; ++ i) {
    if (!value_used_[i] && name_str == argv_[i]) {
      value_used_[i] = true;
      parameters_.back().given = true;
      return true;
    }
  }
  return false;
}

bool CommandLineParser::NamedParameter(const char* name, string* value, bool required, const char* help) {
  CHECK(!sequential_parameter_read_) << "NamedParameters must be read before all sequential parameters";
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = false;
  parameters_.back().required = required;
  if (!required) {
    parameters_.back().default_value = *value;
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  string name_str = name;
  for (int i = 1; i < argc_ - 1; ++ i) {
    if (!value_used_[i] && !value_used_[i + 1] && name_str == argv_[i]) {
      value_used_[i] = true;
      value_used_[i + 1] = true;
      *value = argv_[i + 1];
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::NamedParameter(const char* name, vector<string>* value, char separator, bool required, const char* help) {
  string raw_string;
  if (!NamedParameter(name, &raw_string, required, help)) {
    return false;
  }
  
  value->clear();
  string::size_type pos = 0;
  while (true) {
    const string::size_type separator_pos = raw_string.find(separator, pos);
    if (separator_pos == string::npos) {
      value->push_back(raw_string.substr(pos));
      break;
    } else {
      value->push_back(raw_string.substr(pos, separator_pos - pos));
      pos = separator_pos + 1;
    }
  }
  
  return true;
}

bool CommandLineParser::NamedPathParameter(const char* name, string* value, bool required, const char* help) {
  CHECK(!sequential_parameter_read_) << "NamedParameters must be read before all sequential parameters";
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = false;
  parameters_.back().required = required;
  if (!required) {
    parameters_.back().default_value = *value;
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  string name_str = name;
  for (int i = 1; i < argc_ - 1; ++ i) {
    if (!value_used_[i] && !value_used_[i + 1] && name_str == argv_[i]) {
      value_used_[i] = true;
      value_used_[i + 1] = true;
      *value = argv_[i + 1];
      if (value->size() >= 7 && value->substr(0, 7) == "file://") {
        *value = value->substr(7);
      }
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::NamedPathParameter(const char* name, vector<string>* value, char separator, bool required, const char* help) {
  string raw_string;
  if (!NamedParameter(name, &raw_string, required, help)) {
    return false;
  }
  
  value->clear();
  string::size_type pos = 0;
  while (true) {
    const string::size_type separator_pos = raw_string.find(separator, pos);
    if (separator_pos == string::npos) {
      string item = raw_string.substr(pos);
      if (item.size() >= 7 && item.substr(0, 7) == "file://") {
        item = item.substr(7);
      }
      value->push_back(item);
      break;
    } else {
      string item = raw_string.substr(pos, separator_pos - pos);
      if (item.size() >= 7 && item.substr(0, 7) == "file://") {
        item = item.substr(7);
      }
      value->push_back(item);
      pos = separator_pos + 1;
    }
  }
  
  return true;
}

bool CommandLineParser::NamedParameter(const char* name, int* value, bool required, const char* help) {
  CHECK(!sequential_parameter_read_) << "NamedParameters must be read before all sequential parameters";
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = false;
  parameters_.back().required = required;
  if (!required) {
    ostringstream o;
    o << *value;
    parameters_.back().default_value = o.str();
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  string name_str = name;
  for (int i = 1; i < argc_ - 1; ++ i) {
    if (!value_used_[i] && !value_used_[i + 1] && name_str == argv_[i]) {
      value_used_[i] = true;
      value_used_[i + 1] = true;
      *value = atoi(argv_[i + 1]);
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::NamedParameter(const char* name, float* value, bool required, const char* help) {
  CHECK(!sequential_parameter_read_) << "NamedParameters must be read before all sequential parameters";
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = false;
  parameters_.back().required = required;
  if (!required) {
    ostringstream o;
    o << *value;
    parameters_.back().default_value = o.str();
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  string name_str = name;
  for (int i = 1; i < argc_ - 1; ++ i) {
    if (!value_used_[i] && !value_used_[i + 1] && name_str == argv_[i]) {
      value_used_[i] = true;
      value_used_[i + 1] = true;
      *value = atof(argv_[i + 1]);
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::SequentialParameter(string* value, const char* name, bool required, const char* help) {
  sequential_parameter_read_ = true;
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = true;
  parameters_.back().required = required;
  if (!required) {
    parameters_.back().default_value = *value;
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  for (int i = 1; i < argc_; ++ i) {
    if (!value_used_[i]) {
      value_used_[i] = true;
      *value = argv_[i];
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::SequentialPathParameter(string* value, const char* name, bool required, const char* help) {
  sequential_parameter_read_ = true;
  
  parameters_.push_back(Parameter());
  parameters_.back().name = name;
  parameters_.back().is_flag = false;
  parameters_.back().is_sequential = true;
  parameters_.back().required = required;
  if (!required) {
    parameters_.back().default_value = *value;
  }
  parameters_.back().given = false;
  parameters_.back().help = help;
  
  for (int i = 1; i < argc_; ++ i) {
    if (!value_used_[i]) {
      value_used_[i] = true;
      *value = argv_[i];
      if (value->size() >= 7 && value->substr(0, 7) == "file://") {
        *value = value->substr(7);
      }
      parameters_.back().given = true;
      return true;
    }
  }
  if (required) {
    is_input_complete_ = false;
  }
  return false;
}

bool CommandLineParser::HelpRequested() const {
  return help_requested_;
}

bool CommandLineParser::IsInputComplete() const {
  return is_input_complete_;
}

bool CommandLineParser::UnusedParametersGiven() const {
  for (int i = 1; i < argc_; ++ i) {
    if (!value_used_[i]) {
      return true;
    }
  }
  return false;
}

bool CommandLineParser::CheckParameters() const {
  if (HelpRequested()) {
    ShowUsage();
    return false;
  }
  
  if (!IsInputComplete()) {
    std::cout << "The following required parameter(s) are missing:" << std::endl;
    for (usize i = 1; i < parameters_.size(); ++ i) {
      if (parameters_[i].required && !parameters_[i].given) {
        if (parameters_[i].name.empty()) {
          std::cout << "an unnamed sequential parameter" << std::endl;
        } else {
          std::cout << parameters_[i].name << std::endl;
        }
      }
    }
    std::cout << "Call the program with -h or --help to show its usage." << std::endl;
    return false;
  }
  
  if (UnusedParametersGiven()) {
    std::cout << "Unused parameters were given, aborting:" << std::endl;
    for (int i = 1; i < argc_; ++ i) {
      if (!value_used_[i]) {
        std::cout << argv_[i] << std::endl;
      }
    }
    std::cout << "Call the program with -h or --help to show its usage." << std::endl;
    return false;
  }
  
  return true;
}

void CommandLineParser::ShowUsage() const {
  std::cout << "Usage: " << argv_[0];
  
  // Print required named parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (p.required && !p.is_sequential) {
      std::cout << " " << p.name << " value" ;
    }
  }
  
  // Print required sequential parameters.
  int sequential_parameter_counter = 0;
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (p.required && p.is_sequential) {
      ++ sequential_parameter_counter;
      if (p.name.empty()) {
        std::cout << " sequential_parameter_" << sequential_parameter_counter;
      } else {
        std::cout << " " << p.name;
      }
    }
  }
  
  // Print optional named parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (!p.required && !p.is_sequential) {
      if (p.is_flag) {
        std::cout << " [" << p.name << "]" ;
      } else {
        std::cout << " [" << p.name << " value]" ;
      }
    }
  }
  
  // Print optional sequential parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (!p.required && p.is_sequential) {
      ++ sequential_parameter_counter;
      if (p.name.empty()) {
        std::cout << " [sequential_parameter_" << sequential_parameter_counter << "]";
      } else {
        std::cout << " [" << p.name << "]";
      }
    }
  }
  
  
  // Print help texts.
  std::cout << std::endl << std::endl;
  
  // Required named parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (p.required && !p.is_sequential && !p.help.empty()) {
      std::cout << p.name << ": " << p.help << std::endl;
    }
  }
  
  // Required sequential parameters.
  sequential_parameter_counter = 0;
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (p.required && p.is_sequential) {
      ++ sequential_parameter_counter;
      if (!p.help.empty()) {
        if (p.name.empty()) {
          std::cout << "sequential_parameter_" << sequential_parameter_counter << ": " << p.help << std::endl;
        } else {
          std::cout << p.name << ": " << p.help << std::endl;
        }
      }
    }
  }
  
  // Optional named parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (!p.required && !p.is_sequential) {
      if (p.is_flag) {
        if (!p.help.empty()) {
          std::cout << "[" << p.name << "]: " << p.help << std::endl;
        }
      } else {
        if (p.help.empty()) {
          std::cout << "[" << p.name << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]" << std::endl;
        } else {
          std::cout << "[" << p.name << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]: " << p.help << std::endl;
        }
      }
    }
  }
  
  // Optional sequential parameters.
  for (usize i = 0; i < parameters_.size(); ++ i) {
    const Parameter& p = parameters_[i];
    if (!p.required && p.is_sequential) {
      ++ sequential_parameter_counter;
      if (p.name.empty()) {
        if (p.help.empty()) {
          std::cout << "[sequential_parameter_" << sequential_parameter_counter << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]" << std::endl;
        } else {
          std::cout << "[sequential_parameter_" << sequential_parameter_counter << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]: " << p.help << std::endl;
        }
      } else {
        if (p.help.empty()) {
          std::cout << "[" << p.name << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]" << std::endl;
        } else {
          std::cout << "[" << p.name << ", default: " << (p.default_value.empty() ? "\"\"" : p.default_value) << "]: " << p.help << std::endl;
        }
      }
    }
  }
}

}
