use strum::IntoEnumIterator;

use super::*;

#[test]
fn mime_types_display() {
    for mime_type in crate::ImageMimeType::iter() {
        let expected = match mime_type {
            crate::ImageMimeType::JPEG => "image/jpeg",
            crate::ImageMimeType::PNG => "image/png",
        };

        assert_eq!(expected, mime_type.to_string());
    }
}

#[test]
fn vertex_attributes_display() {
    for mut vertex_attribute in VertexAttribute::iter() {
        const TEST_USER_CASE: &str = "aAbBcC1234!";

        if let VertexAttribute::User(v) = &mut vertex_attribute {
            *v = TEST_USER_CASE.into();
        }

        let expected = match vertex_attribute.clone() {
            VertexAttribute::Position => "POSITION".into(),
            VertexAttribute::Normal => "NORMAL".into(),
            VertexAttribute::Tangent => "TANGENT".into(),
            VertexAttribute::TexCoord(i) => format!("TEXCOORD_{i}"),
            VertexAttribute::Colour(i) => format!("COLOR_{i}"),
            VertexAttribute::Joints(i) => format!("JOINTS_{i}"),
            VertexAttribute::Weights(i) => format!("WEIGHTS_{i}"),
            VertexAttribute::User(_) => TEST_USER_CASE.into(),
        };

        assert_eq!(expected, vertex_attribute.to_string());
    }
}
