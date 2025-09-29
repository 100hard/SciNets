    threshold = _SIMILARITY_THRESHOLDS[resolution_type]
    candidate_pairs = _generate_candidate_pairs(records)
    
    print(f"[DEBUG] Processing {len(candidate_pairs)} candidate pairs for {resolution_type}")
    print(f"[DEBUG] Similarity threshold: {threshold}")
    print(f"[DEBUG] Total records: {len(records)}")
    
    merges_count = 0
    for left_id, right_id in candidate_pairs:
        if uf.find(left_id) == uf.find(left_id):
            continue
        left = record_by_id[left_id]
        right = record_by_id[right_id]
        score = _score_pair(left, right, embeddings)
        
        # Debug similar concepts
        if left.name.lower() == right.name.lower() or any(alias.lower() == right.name.lower() for alias in left.aliases) or any(alias.lower() == left.name.lower() for alias in right.aliases):
            print(f"[DEBUG] EXACT MATCH: '{left.name}' vs '{right.name}': score={score:.3f}")
        
        if score >= threshold:
            print(f"[DEBUG] MERGING: '{left.name}' + '{right.name}' (score={score:.3f})")
            uf.union(left_id, right_id)
            merges_count += 1
        elif score > 0.5:  # Show high-scoring pairs that didn't merge
            print(f"[DEBUG] HIGH SCORE (no merge): '{left.name}' vs '{right.name}': score={score:.3f}")
    
    print(f"[DEBUG] Total merges: {merges_count}")

    groups: Dict[UUID, list[UUID]] = defaultdict(list)
    for record in records:
        root = uf.find(record.id)
        groups[root].append(record.id)
    
    print(f"[DEBUG] Final groups: {len(groups)} (started with {len(records)})")
    for group_id, group_members in groups.items():
        if len(group_members) > 1:
            print(f"[DEBUG] Group with {len(group_members)} members: {[record_by_id[member_id].name for member_id in group_members]}")
