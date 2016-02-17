/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
*/

#include "vw_builder.h"
#include "parser.h"
// #include "primitives.h"

namespace VW
{
    VowpalWabbitExampleBuilder::VowpalWabbitExampleBuilder(VowpalWabbit^ vw) :
        m_vw(vw), m_example(nullptr)
    {
        if (vw == nullptr)
            throw gcnew ArgumentNullException("vw");

        m_example = vw->GetOrCreateNativeExample();
    }

    VowpalWabbitExampleBuilder::~VowpalWabbitExampleBuilder()
    {
        this->!VowpalWabbitExampleBuilder();
    }

    VowpalWabbitExampleBuilder::!VowpalWabbitExampleBuilder()
    {
        if (m_example != nullptr)
        {
            // in case CreateExample is not getting called
            delete m_example;

            m_example = nullptr;
        }
    }

    VowpalWabbitExample^ VowpalWabbitExampleBuilder::CreateExample()
    {
        if (m_example == nullptr)
            return nullptr;

        try
        {
            // finalize example
            VW::parse_atomic_example(*m_vw->m_vw, m_example->m_example, false);
            VW::setup_example(*m_vw->m_vw, m_example->m_example);
        }
        CATCHRETHROW

        // hand memory management off to VowpalWabbitExample
        auto ret = m_example;
        m_example = nullptr;

        return ret;
    }

    void VowpalWabbitExampleBuilder::ParseLabel(String^ value)
    {
        if (value == nullptr)
            return;

        auto bytes = System::Text::Encoding::UTF8->GetBytes(value);
        auto valueHandle = GCHandle::Alloc(bytes, GCHandleType::Pinned);

        try
        {
            VW::parse_example_label(*m_vw->m_vw, *m_example->m_example, reinterpret_cast<char*>(valueHandle.AddrOfPinnedObject().ToPointer()));
        }
        CATCHRETHROW
        finally
        { valueHandle.Free();
        }
    }

    VowpalWabbitNamespaceBuilder^ VowpalWabbitExampleBuilder::AddNamespace(Char featureGroup)
    {
        return AddNamespace((Byte)featureGroup);
    }

    VowpalWabbitNamespaceBuilder^ VowpalWabbitExampleBuilder::AddNamespace(Byte featureGroup)
    {
        uint32_t index = featureGroup;
        example* ex = m_example->m_example;

        return gcnew VowpalWabbitNamespaceBuilder(ex->feature_space + index, featureGroup, m_example->m_example);
    }

    VowpalWabbitNamespaceBuilder::VowpalWabbitNamespaceBuilder(features* features,
        unsigned char index, example* example)
        : m_features(features), m_index(index), m_example(example)
    {
    }

    VowpalWabbitNamespaceBuilder::~VowpalWabbitNamespaceBuilder()
    {
        this->!VowpalWabbitNamespaceBuilder();
    }

    VowpalWabbitNamespaceBuilder::!VowpalWabbitNamespaceBuilder()
    {
        if (m_features->size() > 0)
        {
            unsigned char temp = m_index;
            m_example->indices.push_back(temp);
        }
    }

    void VowpalWabbitNamespaceBuilder::AddFeaturesUnchecked(uint64_t weight_index_base, float* begin, float* end)
    {
        for (; begin != end; begin++)
        {
            float x = *begin;
            if (x != 0)
            {
                m_features->values.push_back_unchecked(x);
                m_features->indicies.push_back_unchecked(weight_index_base);
            }
            weight_index_base++;
        }

    }

    void VowpalWabbitNamespaceBuilder::AddFeature(uint64_t weight_index, float x)
    {
        // filter out 0-values
        if (x == 0)
            return;

        m_features->push_back(x, weight_index);
    }

    void VowpalWabbitNamespaceBuilder::PreAllocate(int size)
    {
        m_features->values.resize(m_features->values.end() - m_features->values.begin() + size);
        m_features->indicies.resize(m_features->indicies.end() - m_features->indicies.begin() + size);
    }
}
