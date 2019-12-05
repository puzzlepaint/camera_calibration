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


#pragma once

namespace vis {

// HACK to determine whether class T has the degrees_of_freedom() and / or
// rows() functions. This is used to allow Eigen vector types as State in
// LMOptimizer and getting the variable count using their rows() function in
// this case, while using a function with the better-suited name
// degrees_of_freedom() in other cases. If neither exists, the compile error will
// complain about degrees_of_freedom() missing, not rows(). Sources:
// http://stackoverflow.com/questions/12015195/how-to-call-member-function-only-if-object-happens-to-have-it
// http://stackoverflow.com/questions/29772601/why-is-sfinae-causing-failure-when-there-are-two-functions-with-different-signat
template<typename T>
struct DegreesOfFreedomGetter {
  // NOTE: No function bodies are needed as they are never called.

  // If the member function A_CLASS::degrees_of_freedom exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      degrees_of_freedom_exists(decltype(std::declval<A_CLASS>().degrees_of_freedom())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      degrees_of_freedom_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(degrees_of_freedom_exists<T>(nullptr))
      degrees_of_freedom_exists_result_type;
  // This will have the value true or false depending on the result.
  static constexpr bool degrees_of_freedom_exists_result =
      degrees_of_freedom_exists_result_type::value;
  
  // If the member function A_CLASS::rows exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      rows_exists(decltype(std::declval<A_CLASS>().rows())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      rows_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(rows_exists<T>(nullptr)) rows_exists_result_type;
  // This will have the value true or false depending on the result.
  static constexpr bool rows_exists_result = rows_exists_result_type::value;
  
  // This is called if both rows() and degrees_of_freedom() exist.
  static int _eval(const T& object, std::true_type, std::true_type) {
    return object.degrees_of_freedom();
  }
  
  // This is called if only degrees_of_freedom() exists.
  static int _eval(const T& object, std::true_type, std::false_type) {
    return object.degrees_of_freedom();
  }
  
  // This is called if only rows() exists.
  static int _eval(const T& object, std::false_type, std::true_type) {
    return object.rows();
  }
  
  // This is called for otherwise unmatched arguments: neither rows() nor
  // degrees_of_freedom() exist.
  static int _eval(const T& object, ...) {
    // Will raise a compile error about object.degrees_of_freedom() missing.
    return object.degrees_of_freedom();
  }

  // Delegates to the function whose parameter types fit the types of
  // the results.
  static int eval(const T& object) {
    return _eval(object,
                 degrees_of_freedom_exists_result_type(),
                 rows_exists_result_type());
  }
};

// HACK to determine whether class T has the is_reversible() function, or
// rows(). Assumes that the class is reversible if it has rows().
template<typename T>
struct IsReversibleGetter {
  // NOTE: No function bodies are needed as they are never called.

  // If the member function A_CLASS::is_reversible exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      is_reversible_exists(decltype(std::declval<A_CLASS>().is_reversible())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      is_reversible_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(is_reversible_exists<T>(nullptr))
      is_reversible_exists_result_type;
  // This will have the value true or false depending on the result.
  static constexpr bool is_reversible_exists_result =
      is_reversible_exists_result_type::value;
  
  // If the member function A_CLASS::rows exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      rows_exists(decltype(std::declval<A_CLASS>().rows())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      rows_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(rows_exists<T>(nullptr)) rows_exists_result_type;
  // This will have the value true or false depending on the result.
  static constexpr bool rows_exists_result = rows_exists_result_type::value;
  
  // This is called if both rows() and is_reversible() exist.
  static constexpr bool _eval(std::true_type, std::true_type) {
    return T::is_reversible();
  }
  
  // This is called if only is_reversible() exists.
  static constexpr bool _eval(std::true_type, std::false_type) {
    return T::is_reversible();
  }
  
  // This is called if only rows() exists.
  static constexpr bool _eval(std::false_type, std::true_type) {
    return true;
  }
  
  // This is called for otherwise unmatched arguments: neither rows() nor
  // is_reversible() exist.
  static constexpr bool _eval(...) {
    return false;
  }
  
  // Delegates to the function whose parameter types fit the types of
  // the results.
  static constexpr bool eval() {
    return _eval(is_reversible_exists_result_type(),
                 rows_exists_result_type());
  }
};

}
