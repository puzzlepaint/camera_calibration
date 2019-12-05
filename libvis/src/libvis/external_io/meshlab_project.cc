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

#include "libvis/external_io/meshlab_project.h"

#include <tinyxml2/tinyxml2.h>

using namespace tinyxml2;

namespace vis {

bool ReadMeshLabProject(const std::string& project_file_path, MeshLabMeshInfoVector* meshes) {
  XMLDocument doc;
  if (doc.LoadFile(project_file_path.c_str()) != XML_SUCCESS) {
    return false;
  }
  
  XMLElement* xml_meshlabproject = doc.FirstChildElement("MeshLabProject");
  if (!xml_meshlabproject) {
    return false;
  }
  XMLElement* xml_mesh_group = xml_meshlabproject->FirstChildElement("MeshGroup");
  if (!xml_mesh_group) {
    return false;
  }
  XMLElement* xml_mlmesh = xml_mesh_group->FirstChildElement("MLMesh");
  while (xml_mlmesh) {
    MeshLabMeshInfo new_mesh;
    if (xml_mlmesh->Attribute("label")) {
      new_mesh.label = xml_mlmesh->Attribute("label");
    }
    if (xml_mlmesh->Attribute("filename")) {
      new_mesh.filename = xml_mlmesh->Attribute("filename");
    }
    
    XMLElement* xml_mlmatrix44 = xml_mlmesh->FirstChildElement("MLMatrix44");
    if (xml_mlmatrix44) {
      std::string mlmatrix44_text = xml_mlmatrix44->GetText();
      std::istringstream mlmatrix44_stream(mlmatrix44_text);
      Mat4f& M = new_mesh.global_tr_mesh;
      mlmatrix44_stream >> M(0, 0) >> M(0, 1) >> M(0, 2) >> M(0, 3);
      mlmatrix44_stream >> M(1, 0) >> M(1, 1) >> M(1, 2) >> M(1, 3);
      mlmatrix44_stream >> M(2, 0) >> M(2, 1) >> M(2, 2) >> M(2, 3);
      mlmatrix44_stream >> M(3, 0) >> M(3, 1) >> M(3, 2) >> M(3, 3);
    } else {
      // Default-constructed transformation will be identity.
    }
    
    meshes->push_back(new_mesh);
    xml_mlmesh = xml_mlmesh->NextSiblingElement("MLMesh");
  }
  
  return true;
}

bool WriteMeshLabProject(const std::string& project_file_path, const MeshLabMeshInfoVector& meshes) {
  XMLDocument doc;

  XMLElement* xml_meshlabproject = doc.NewElement("MeshLabProject");
  doc.InsertEndChild(xml_meshlabproject);
  
  XMLElement* xml_meshgroup = doc.NewElement("MeshGroup");
  xml_meshlabproject->InsertEndChild(xml_meshgroup);
  
  for (const MeshLabMeshInfo& mesh : meshes) {
    XMLElement* xml_mlmesh = doc.NewElement("MLMesh");
    xml_mlmesh->SetAttribute("label", mesh.label.c_str());
    xml_mlmesh->SetAttribute("filename", mesh.filename.c_str());
    xml_meshgroup->InsertEndChild(xml_mlmesh);
    
    XMLElement* xml_mlmatrix44 = doc.NewElement("MLMatrix44");
    std::ostringstream mlmatrix44_stream;
    mlmatrix44_stream << std::endl;
    // The spaces at the end are important. If omitted, MeshLab will crash when
    // opening the file.
    const Mat4f& M = mesh.global_tr_mesh;
    mlmatrix44_stream << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << " " << M(0, 3) << " " << std::endl;
    mlmatrix44_stream << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << " " << M(1, 3) << " " << std::endl;
    mlmatrix44_stream << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << " " << M(2, 3) << " " << std::endl;
    mlmatrix44_stream << M(3, 0) << " " << M(3, 1) << " " << M(3, 2) << " " << M(3, 3) << " " << std::endl;
    xml_mlmatrix44->SetText(mlmatrix44_stream.str().c_str());
    xml_mlmesh->InsertEndChild(xml_mlmatrix44);
  }
  
  return (doc.SaveFile(project_file_path.c_str()) == tinyxml2::XML_NO_ERROR);
}

}
