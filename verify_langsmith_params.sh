#!/bin/bash

echo "Checking all agents for LangSmith parameters..."

for agent in src/agents/{finance,market,goal,news,portfolio,tax}_agent.py; do
    echo ""
    echo "Checking: $(basename $agent)"
    
    if grep -q "reference_context=" "$agent"; then
        echo "  ✅ Has reference_context"
    else
        echo "  ❌ Missing reference_context"
    fi
    
    if grep -q "reference_outputs=" "$agent"; then
        echo "  ✅ Has reference_outputs"
    else
        echo "  ❌ Missing reference_outputs"
    fi
done
