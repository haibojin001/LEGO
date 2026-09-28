from typing import List, Union

from cover_agent.lsp_logic.multilspy.lsp_protocol_handler import lsp_types


class LspRequest:
    def __init__(self, send_request):
        self.send_request = send_request

    async def implementation(
        self, params: lsp_types.ImplementationParams
    ) -> Union["lsp_types.Definition", List["lsp_types.LocationLink"], None]:
        return await self.send_request("textDocument/implementation", params)

    async def type_definition(
        self, params: lsp_types.TypeDefinitionParams
    ) -> Union["lsp_types.Definition", List["lsp_types.LocationLink"], None]:
        return await self.send_request("textDocument/typeDefinition", params)

    async def document_color(
        self, params: lsp_types.DocumentColorParams
    ) -> List["lsp_types.ColorInformation"]:
        return await self.send_request("textDocument/documentColor", params)

    async def color_presentation(
        self, params: lsp_types.ColorPresentationParams
    ) -> List["lsp_types.ColorPresentation"]:
        return await self.send_request("textDocument/colorPresentation", params)

    async def folding_range(
        self, params: lsp_types.FoldingRangeParams
    ) -> Union[List["lsp_types.FoldingRange"], None]:
        return await self.send_request("textDocument/foldingRange", params)

    async def declaration(
        self, params: lsp_types.DeclarationParams
    ) -> Union["lsp_types.Declaration", List["lsp_types.LocationLink"], None]:
        return await self.send_request("textDocument/declaration", params)

    async def selection_range(
        self, params: lsp_types.SelectionRangeParams
    ) -> Union[List["lsp_types.SelectionRange"], None]:
        return await self.send_request("textDocument/selectionRange", params)

    async def prepare_call_hierarchy(
        self, params: lsp_types.CallHierarchyPrepareParams
    ) -> Union[List["lsp_types.CallHierarchyItem"], None]:
        return await self.send_request("textDocument/prepareCallHierarchy", params)

    async def incoming_calls(
        self, params: lsp_types.CallHierarchyIncomingCallsParams
    ) -> Union[List["lsp_types.CallHierarchyIncomingCall"], None]:
        return await self.send_request("callHierarchy/incomingCalls", params)

    async def outgoing_calls(
        self, params: lsp_types.CallHierarchyOutgoingCallsParams
    ) -> Union[List["lsp_types.CallHierarchyOutgoingCall"], None]:
        return await self.send_request("callHierarchy/outgoingCalls", params)

    async def semantic_tokens_full(
        self, params: lsp_types.SemanticTokensParams
    ) -> Union["lsp_types.SemanticTokens", None]:
        return await self.send_request("textDocument/semanticTokens/full", params)

    async def semantic_tokens_delta(
        self, params: lsp_types.SemanticTokensDeltaParams
    ) -> Union["lsp_types.SemanticTokens", "lsp_types.SemanticTokensDelta", None]:
        return await self.send_request(
            "textDocument/semanticTokens/full/delta", params
        )

    async def semantic_tokens_range(
        self, params: lsp_types.SemanticTokensRangeParams
    ) -> Union["lsp_types.SemanticTokens", None]:
        return await self.send_request("textDocument/semanticTokens/range", params)

    async def linked_editing_range(
        self, params: lsp_types.LinkedEditingRangeParams
    ) -> Union["lsp_types.LinkedEditingRanges", None]:
        return await self.send_request("textDocument/linkedEditingRange", params)

    async def will_create_files(
        self, params: lsp_types.CreateFilesParams
    ) -> Union["lsp_types.WorkspaceEdit", None]:
        return await self.send_request("workspace/willCreateFiles", params)

    async def will_rename_files(
        self, params: lsp_types.RenameFilesParams
    ) -> Union["lsp_types.WorkspaceEdit", None]:
        return await self.send_request("workspace/willRenameFiles", params)

    async def will_delete_files(
        self, params: lsp_types.DeleteFilesParams
    ) -> Union["lsp_types.WorkspaceEdit", None]:
        return await self.send_request("workspace/willDeleteFiles", params)

    async def moniker(
        self, params: lsp_types.MonikerParams
    ) -> Union[List["lsp_types.Moniker"], None]:
        return await self.send_request("textDocument/moniker", params)

    async def prepare_type_hierarchy(
        self, params: lsp_types.TypeHierarchyPrepareParams
    ) -> Union[List["lsp_types.TypeHierarchyItem"], None]:
        return await self.send_request("textDocument/prepareTypeHierarchy", params)

    async def type_hierarchy_supertypes(
        self, params: lsp_types.TypeHierarchySupertypesParams
    ) -> Union[List["lsp_types.TypeHierarchyItem"], None]:
        return await self.send_request("typeHierarchy/supertypes", params)

    async def type_hierarchy_subtypes(
        self, params: lsp_types.TypeHierarchySubtypesParams
    ) -> Union[List["lsp_types.TypeHierarchyItem"], None]:
        return await self.send_request("typeHierarchy/subtypes", params)

    async def inline_value(
        self, params: lsp_types.InlineValueParams
    ) -> Union[List["lsp_types.InlineValue"], None]:
        return await self.send_request("textDocument/inlineValue", params)

    async def inlay_hint(
        self, params: lsp_types.InlayHintParams
    ) -> Union[List["lsp_types.InlayHint"], None]:
        return await self.send_request("textDocument/inlayHint", params)

    async def inlay_hint_resolve(
        self, params: lsp_types.InlayHint
    ) -> "lsp_types.InlayHint":
        return await self.send_request("inlayHint/resolve", params)

    async def document_diagnostic(
        self, params: lsp_types.DocumentDiagnosticParams
    ) -> "lsp_types.DocumentDiagnosticReport":
        return await self.send_request("textDocument/diagnostic", params)

    async def workspace_diagnostic(
        self, params: lsp_types.WorkspaceDiagnosticParams
    ) -> "lsp_types.WorkspaceDiagnosticReport":
        return await self.send_request("workspace/diagnostic", params)