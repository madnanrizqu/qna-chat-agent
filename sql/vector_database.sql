CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL,
    category TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(3072),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 3
)
RETURNS TABLE (id UUID, content TEXT, category TEXT, similarity FLOAT)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT d.id, d.content, d.category,
            1 - (d.embedding <=> query_embedding) AS
similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) >
match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;